import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{Encoding, EncodingRegistry, EncodingType, IntArrayList}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.ops.transforms.Transforms
import org.slf4j.LoggerFactory
import java.util
import scala.jdk.CollectionConverters.asScalaIteratorConverter

object SlidingWindowWithPositionalEmbedding {

  private val logger = LoggerFactory.getLogger(SparkLLMTraining.getClass)


  // Broadcast variables for distributed access across workers
  private var embeddingMapBroadcast: Broadcast[Map[Int, INDArray]] = _
  private var embeddingDimBroadcast: Broadcast[Int] = _
  private var oovBroadcast: Broadcast[INDArray] = _
  private var vocabSizeBroadcast: Broadcast[Int] = _

  private var encodingType: EncodingType = getEncodingType("r50k_base")
  private val registry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()
  private val encoding: Encoding = registry.getEncoding(encodingType)

  def initEmbeddingMap(sc: SparkContext, directoryPath: String, encodeType: String = "default"): Unit = {
    // Load embeddings and create the map
    val embeddingMap = FileUtil.loadEmbeddings(sc, directoryPath)

    if (embeddingMap.nonEmpty) {
      // Set embeddingDim to the length of the first embedding vector
      val embeddingDim = embeddingMap.head._2.length.toInt

      // Set oov vector to predefined random embedding vector
      Nd4j.getRandom.setSeed(12345L)
      val oov = Transforms.unitVec(Nd4j.rand(DataType.DOUBLE, 1, embeddingDim))

      // Broadcast embedding map, dimension, and OOV embedding
      embeddingMapBroadcast = sc.broadcast(embeddingMap)
      embeddingDimBroadcast = sc.broadcast(embeddingDim)
      oovBroadcast = sc.broadcast(oov)
    } else {
      logger.warn("WARNING: Embedding map is empty; embeddingDim and OOV not set")
    }

    // Set vocab size based on encoding type
    if (encodeType != "default") {
      encodingType = getEncodingType(encodeType)
      logger.info(s"Encoding type set to $encodeType. If invalid, defaulting to r50k_base.")
    }
    val vocabSize = if (encodingType == EncodingType.CL100K_BASE) 100000 else 50256
    vocabSizeBroadcast = sc.broadcast(vocabSize)

    logger.info("Embeddings map broadcasted and ready for distributed access")
  }

  def getVocabSize: Int = vocabSizeBroadcast.value

  def getEmbeddingDim: Int = embeddingDimBroadcast.value

  def getEmbeddingMap: Map[Int, INDArray] = embeddingMapBroadcast.value

  def getOovEmbedding: INDArray = oovBroadcast.value

  // Ensure cleanup of broadcast variables
  def clearBroadcasts(): Unit = {
    if (embeddingMapBroadcast != null) embeddingMapBroadcast.destroy()
    if (embeddingDimBroadcast != null) embeddingDimBroadcast.destroy()
    if (oovBroadcast != null) oovBroadcast.destroy()
    if (vocabSizeBroadcast != null) vocabSizeBroadcast.destroy()
  }

  // Updated function to use an existing RDD instead of creating a new one
  def createSlidingWindowsWithPositionalEmbedding(tokenString: String, windowSize: Int): Iterator[DataSet] = {

    // Encode tokens based on the broadcasted encoding
    val tokens: Array[Int] = encodeTokens(tokenString)

    // Use sliding to create windows directly
    tokens.sliding(windowSize + 1).map { window: Array[Int] =>

      // Extract input window (first `windowSize` tokens) and target token (last token)
      val inputWindow = window.take(windowSize)
      val targetIndex = window.last

      // Use broadcasted embedding map to embed the tokens
      val features: INDArray = tokenizeAndEmbed(inputWindow)

      // Initialize the one-hot encoded target vector using the broadcast vocab size
      val vocabSize = vocabSizeBroadcast.value
      val label = Nd4j.zeros(1, vocabSize, windowSize)

      // Set target index for each timestep
      require(targetIndex < vocabSize, s"Target index $targetIndex out of bounds for vocab size $vocabSize")
      for (t <- 0 until windowSize) {
        label.putScalar(Array(0, targetIndex, t), 1.0)
      }

      new DataSet(features, label)
    }
  }



  def batchSlidingWindows(dataRDD: RDD[DataSet], batchSize: Int): RDD[DataSet] = {

    // Group data into batches and process each batch in partitions
    val batchedRDD = dataRDD
      .zipWithIndex()  // Add index to each DataSet to help with batching
      .map { case (dataSet, idx) => (idx / batchSize, dataSet) }  // Assign each dataset to a batch
      .groupByKey()  // Group by batch index
      .mapPartitions { batches =>
        val batchedDataList = new util.ArrayList[DataSet]()

        for ((_, batch) <- batches) {
          // Create an ArrayList for the current batch
          val dataSetBatch = new util.ArrayList[DataSet]()
          batch.foreach(dataSetBatch.add)

          // Merge the batch datasets
          val batchDataSet = DataSet.merge(dataSetBatch)

          // Check the shape of merged features to confirm initial arrangement
          val originalShape = batchDataSet.getFeatures.shape()

          // Calculate windowSize based on original shape and batch size
          val windowSize: Int = (originalShape(0) / batchSize).toInt

          // Reshape and transpose to ensure embeddings are columns
          // Step 1: Reshape to [windowSize, batchSize, embeddingDim]
          // Step 2: Permute dimensions to [batchSize, embeddingDim, windowSize]
          val features = batchDataSet.getFeatures.reshape(windowSize, batchSize, embeddingDimBroadcast.value).permute(1, 2, 0)

          // Reshape labels as needed
          val labels = batchDataSet.getLabels.reshape(batchSize, vocabSizeBroadcast.value, windowSize)

          // Update the dataset with reshaped features and labels
          batchDataSet.setFeatures(features)
          batchDataSet.setLabels(labels)

          // Add the batched DataSet to the list
          batchedDataList.add(batchDataSet)
        }

        // Convert the iterator of batches into an iterator of individual DataSets
        batchedDataList.iterator().asScala
      }

    batchedRDD
  }


  def encodeTokens(sentence: String): Array[Int] = {
      encoding.encode(sentence).toArray
  }

  def tokenizeAndEmbed(tokens: Array[Int]): INDArray = {

    // Fetch embeddings for each token
    val embeddings = tokens.map { tokenId =>
      val embedding = embeddingMapBroadcast.value.get(tokenId) match {
        case Some(embeddingArray) =>
          embeddingArray
        case None =>
          oovBroadcast.value
      }

      // Ensure all embeddings are 2D: reshape [100] to [1,100]
      if (embedding.shape().length == 1 && embedding.shape()(0) == embeddingDimBroadcast.value) {
        embedding.reshape(1, embeddingDimBroadcast.value)  // Reshape 1D array to 2D
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
    val normalizedPositionAwareEmbedding = Nd4j.create(tokens.length, embeddingDimBroadcast.value)

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

    val positionalEncoding = Nd4j.zeros(windowSize, embeddingDimBroadcast.value)
    for (pos <- 0 until windowSize) {
      var i = 0
      while (i < embeddingDimBroadcast.value) {
        val angle = pos / Math.pow(10000, (2.0 * i) / embeddingDimBroadcast.value)
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

