import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{Encoding, EncodingRegistry, EncodingType, IntArrayList}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.util.LongAccumulator
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.ops.transforms.Transforms

import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters.bufferAsJavaListConverter

object SlidingWindowWithPositionalEmbedding {

  // vars since we are initializing these during runtime
  private var embeddingMap: Map[Int, INDArray] = Map()   // Map of token ids to their vector embeddings
  private var embeddingDim: Int = 0 // Number of dimensions of each embedding

  private var encodingType: EncodingType = getEncodingType("r50k_base")
  private val registry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()
  private val encoding: Encoding = registry.getEncoding(encodingType)
  private var vocabSize: Int = 50256

  def initEmbeddingMap(sc: SparkContext, directoryPath: String, encodeType: String = "default"): Unit = {
    // Load embeddings and create the map
    embeddingMap = FileUtil.loadEmbeddings(sc, directoryPath)

    if (embeddingMap.nonEmpty) {
      // Set embeddingDim to the length of the first embedding vector
      embeddingDim = embeddingMap.head._2.length.toInt
    } else {
      throw new IllegalStateException("Embedding map empty")
    }

    // Set vocab size based on encoding type
    if (encodeType != "default") {
      encodingType = getEncodingType(encodeType)
    }
    vocabSize = if (encodingType == EncodingType.CL100K_BASE) 100000 else 50256

  }

  def getVocabSize: Int = vocabSize

  def getEmbeddingDim: Int = embeddingDim

  def getEmbeddingMap: Map[Int, INDArray] = embeddingMap

  // Updated function to use an existing RDD instead of creating a new one
  def createSlidingWindowsWithPositionalEmbedding(tokenString: String, windowSize: Int, vocabSize: Int, embeddingDim: Int, embeddingMap: Broadcast[Map[Int, INDArray]]): Iterator[DataSet] = {

    // Encode tokens based on the broadcasted encoding
    val tokens: Array[Int] = encodeTokens(tokenString)

    if (tokens.length <= windowSize) return Iterator.empty

    // Use sliding to create windows directly
    tokens.sliding(windowSize + 1).map { window: Array[Int] =>

      // Extract input window (first `windowSize` tokens) and target token (last token)
      val inputWindow = window.take(windowSize)

      // Use broadcasted embedding map to embed the tokens
      val features: INDArray = tokenizeAndEmbed(inputWindow, embeddingMap.value, embeddingDim)

      // Initialize the one-hot encoded target vector using the broadcast vocab size
      val label = Nd4j.zeros(1, vocabSize, windowSize)

      // Set target index for each timestep
      for (t <- 0 until windowSize) {
        label.putScalar(Array(0, window(t + 1), t), 1.0)
      }

      new DataSet(features, label)
    }
  }

  // Batches sliding window DataSet objects into larger DataSets of a specified batch size.
  def batchSlidingWindows(iter: Iterator[DataSet], batchSize: Int, embeddingDim: Int, vocabSize: Int): Iterator[DataSet] = {
    val batchedList = ArrayBuffer[DataSet]()
    val batchBuffer = ArrayBuffer[DataSet]()

    iter.foreach { dataSet =>
      // Add each DataSet to the current batch buffer
      batchBuffer += dataSet

      if (batchBuffer.size == batchSize) {
        val batchDataSet = DataSet.merge(batchBuffer.asJava)
        val originalShape = batchDataSet.getFeatures.shape()
        val windowSize: Int = (originalShape(0) / batchSize).toInt

        val reshapedFeatures = batchDataSet.getFeatures.reshape(windowSize, batchSize, embeddingDim).permute(1, 2, 0)
        val reshapedLabels = batchDataSet.getLabels.reshape(batchSize, vocabSize, windowSize)

        batchDataSet.setFeatures(reshapedFeatures)
        batchDataSet.setLabels(reshapedLabels)

        batchedList += batchDataSet
        batchBuffer.clear()
      }
    }
    batchedList.iterator
  }


  def encodeTokens(sentence: String): Array[Int] = {
      encoding.encode(sentence).toArray
  }

  def tokenizeAndEmbed(tokens: Array[Int], embeddingMap: Map[Int, INDArray], embeddingDim: Int): INDArray = {

    if (embeddingMap == null) {
      throw new IllegalStateException("Embedding map is null. Cannot proceed with token embedding.")
    }

    // Fetch embeddings for each token
    val embeddings = tokens.map { tokenId =>
      val embedding = embeddingMap.get(tokenId) match {
        case Some(embeddingArray) =>
          embeddingArray
        case None =>
          Nd4j.getRandom.setSeed(12345L)
          Transforms.unitVec(Nd4j.rand(DataType.DOUBLE, 1, embeddingDim))  // Default to OOV vector if the token is not found
      }

      // Ensure embedding is not null before accessing its shape
      if (embedding == null) {
        throw new IllegalStateException(s"Embedding for token ID $tokenId is null, and OOV is also null.")
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
    val positionalEmbeddings: INDArray = computePositionalEmbedding(tokens.length, embeddingDim)

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
  def computePositionalEmbedding(windowSize: Int, embeddingDim: Int): INDArray = {

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

