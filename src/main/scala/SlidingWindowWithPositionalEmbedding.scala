import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{Encoding, EncodingRegistry, EncodingType, IntArrayList}
import org.apache.spark.SparkContext
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms
import org.slf4j.LoggerFactory

import java.io.File
import java.util
import scala.io.Source
import scala.collection.mutable
import scala.jdk.CollectionConverters.asScalaBufferConverter
import scala.util.Using

object SlidingWindowWithPositionalEmbedding {

  private val logger = LoggerFactory.getLogger(SparkLLMTraining.getClass)


  // vars since we are initializing these during runtime
  private var embeddingMap: Map[Int, INDArray] = Map()   // Map of token ids to their vector embeddings
  private var embeddingDim: Int = 0 // Number of dimensions of each embedding
  private var oov: INDArray = _ // Out of vocabulary embedding generated

  private var encodingType: EncodingType = getEncodingType("r50k_base")
  private val registry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()
  private val encoding: Encoding = registry.getEncoding(encodingType)
  private var vocabSize: Int = 50256

  def initEmbeddingMap(sc: SparkContext, directoryPath: String, encodeType: String = "default"): Unit = {

    // Convert the mutable map to an immutable one
    embeddingMap = FileUtil.loadEmbeddings(sc, directoryPath)

    if (embeddingMap.nonEmpty) {
      // Set embeddingDim to the length of the first embedding vector
      embeddingDim = embeddingMap.head._2.length.toInt

      // Set oov vector to predefined random embedding vector
      Nd4j.getRandom.setSeed(12345L)
      oov = Transforms.unitVec(Nd4j.rand(DataType.DOUBLE, 1, embeddingDim))
    } else {
      logger.warn("WARNING: Embedding map empty, embeddingDim and OOV not set")
    }

    if (encodeType != "default") {
      encodingType = getEncodingType(encodeType)
      if (encodingType == EncodingType.CL100K_BASE) {
        vocabSize = 100000
      }
      logger.info(s"Encoding type set to ${encodeType}. If invalid, will default to r50k_base")
    }

    logger.info("Embeddings map created")
  }

  def getVocabSize: Int = vocabSize

  def getEmbeddingDim: Int = embeddingDim

  def getMap: Map[Int, INDArray] = embeddingMap

  // Create sliding windows for inputs and targets with positional embeddings
  def createSlidingWindowsWithPositionalEmbedding(tokenString: String, windowSize: Int): util.ArrayList[DataSet] = {

    val tokens: Array[Int] = encodeTokens(tokenString)
    val dataSetList: util.ArrayList[DataSet] = new util.ArrayList[DataSet]()

    for (i <- 0 until tokens.length - (windowSize)) {

      // Extract the input window (windowSize tokens)
      val inputWindow = new Array[Int](windowSize)
      System.arraycopy(tokens, i, inputWindow, 0, windowSize)

      // Convert input tokens into normalized, positional embeddings for the features
      val features: INDArray = tokenizeAndEmbed(inputWindow)

      // Initialize the one-hot encoded target vector with shape [1, vocabSize, sequenceLength]
      val label = Nd4j.zeros(1, vocabSize, windowSize) // Shape: [minibatch, vocabSize, sequenceLength]

      // Set the target index for each timestep
      val targetIndex = tokens(i + windowSize)  // Assuming this is the token index for the target
      for (t <- 0 until windowSize) {
        if(targetIndex > vocabSize) throw new IndexOutOfBoundsException("yo mama: " + targetIndex + " " + vocabSize)
        label.putScalar(Array(0, targetIndex, t), 1.0) // Apply the same target across all timesteps
      }

      dataSetList.add(new DataSet(features, label))
    }
    dataSetList
  }

  def batchSlidingWindows(data: util.ArrayList[DataSet], batchSize: Int): util.ArrayList[DataSet] = {
    val batchedData = new util.ArrayList[DataSet]()
    val numBatches = data.size() / batchSize // Only full batches are considered

    // Create full batches
    for (i <- 0 until numBatches) {
      val batch = new util.ArrayList[DataSet]()
      for (j <- 0 until batchSize) {
        batch.add(data.get(i * batchSize + j))
      }

      val batchDataSet = DataSet.merge(batch)

      // Check the shape of merged features to confirm initial arrangement
      val originalShape = batchDataSet.getFeatures.shape()

      // Calculate windowSize based on original shape and batch size
      val windowSize: Int = (originalShape(0) / batchSize).toInt

      // Reshape and transpose to ensure embeddings are columns
      // Step 1: Reshape to [windowSize, batchSize, embeddingDim]
      // Step 2: Permute dimensions to [batchSize, embeddingDim, windowSize]
      val features = batchDataSet.getFeatures.reshape(windowSize, batchSize, embeddingDim).permute(1, 2, 0)

      // Reshape labels as needed
      val labels = batchDataSet.getLabels.reshape(batchSize, vocabSize, windowSize)

      if(i == 0) {
        println("feature pre reshape: " + batchDataSet.getFeatures)
      }

      // Update the dataset with reshaped features and labels
      batchDataSet.setFeatures(features)
      batchDataSet.setLabels(labels)

      batchedData.add(batchDataSet)
    }


    // Any remaining data is ignored (dropped) since we don't process short batches.

    batchedData // Return the batched data
  }

  def encodeTokens(sentence: String): Array[Int] = {
      encoding.encode(sentence).toArray
  }

  def tokenizeAndEmbed(tokens: Array[Int]): INDArray = {

    // Fetch embeddings for each token
    val embeddings = tokens.map { tokenId =>
      val embedding = embeddingMap.get(tokenId) match {
        case Some(embeddingArray) =>
          embeddingArray
        case None =>
          oov
      }

      // Ensure all embeddings are 2D: reshape [100] to [1,100]
      if (embedding.shape().length == 1 && embedding.shape()(0) == embeddingDim) {
        embedding.reshape(1, embeddingDim)  // Reshape 1D array to 2D
      } else {
        embedding
      }
    }

    // Stack all embeddings into a single INDArray
    if (embeddings.isEmpty)
      throw new IllegalStateException("Embeddings could not be created for token string")

    val stackedEmbeddings = Nd4j.vstack(embeddings: _*)  // Stack embeddings vertically to create a matrix [num_tokens x embeddingDim]

    // Compute positional embeddings
    val positionalEmbeddings: INDArray = computePositionalEmbedding(tokens.length)

    // Add positional embeddings to the word embeddings
    val positionAwareEmbedding: INDArray = stackedEmbeddings.add(positionalEmbeddings)

    // Create an empty INDArray to store the normalized embeddings
    val normalizedPositionAwareEmbedding = Nd4j.create(tokens.length, embeddingDim)

    // Normalize each row independently
    for (i <- 0 until tokens.length) {
      val rowVector = positionAwareEmbedding.getRow(i)  // Get the row as an INDArray
      val normalizedRow = Transforms.unitVec(rowVector) // Normalize the row to unit length
      normalizedPositionAwareEmbedding.putRow(i, normalizedRow) // Insert back into the result matrix
    }

    normalizedPositionAwareEmbedding
  }


  def translateIndex(index: Int): String = {
    val i: IntArrayList = new IntArrayList()
    i.add(index)
    encoding.decode(i)
  }




  // Compute sinusoidal positional embeddings for a given window size
  def computePositionalEmbedding(windowSize: Int): INDArray = {

    val positionalEncoding = Nd4j.zeros(windowSize, embeddingDim)
    for (pos <- 0 until windowSize) {
      var i = 0
      while (i < embeddingDim) {
        val angle = pos / Math.pow(10000, (2.0 * i) / embeddingDim)
        positionalEncoding.putScalar(Array[Int](pos, i), Math.sin(angle))
        positionalEncoding.putScalar(Array[Int](pos, i + 1), Math.cos(angle))

        i += 2
      }
    }
    positionalEncoding
  }

  private def getEncodingType(encoding: String): EncodingType = {
    encoding match {
      case "cl100k_base" => EncodingType.CL100K_BASE
      case "r50k_base"   => EncodingType.R50K_BASE
      case "p50k_base"   => EncodingType.P50K_BASE
      case "p50k_edit"   => EncodingType.P50K_EDIT
      case _             => EncodingType.R50K_BASE
    }
  }
}

