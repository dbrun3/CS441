import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingRegistry
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.DataSet
import org.slf4j.LoggerFactory

import java.io.File
import java.util
import scala.io.Source
import scala.collection.mutable
import scala.util.Using

object SlidingWindowWithPositionalEmbedding {

  private val logger = LoggerFactory.getLogger(SparkLLMTraining.getClass)


  // vars since we are initializing these during runtime
  private var embeddingMap: Map[Int, Array[Double]] = Map()   // Map of token ids to their vector embeddings
  private var embeddingDim: Int = 0 // Number of dimensions of each embedding
  private var oov: INDArray = _ // Out of vocabulary embedding generated

  private var encodingType: EncodingType = getEncodingType("r50k_base")
  private val registry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()
  private val encoding: Encoding = registry.getEncoding(encodingType)

  def initEmbeddingMap(directoryPath: String, encodeType: String = "default"): Unit = {
    val embeddingMapBuilder = mutable.Map[Int, Array[Double]]()

    // Get all files in the specified directory
    val dir = new File(directoryPath)
    if (dir.exists && dir.isDirectory) {
      val files = dir.listFiles.filter(_.isFile).toList

      // Iterate over each file in the directory
      files.foreach { file =>
        // Use `Using` to safely handle file resource management
        Using(Source.fromFile(file)) { source =>
          for (line <- source.getLines()) {
            // Regex to extract the ID and the embedding vector
            val pattern = """word:\s+\S+\s+id:\s+(\d+)\s+freq:\s+\d+\s+\[([^\]]+)\]""".r //word: blah id: 123 freq: 123 [1, 2, 3]

            line match {
              case pattern(idString, embeddingString) =>
                // Parse the ID
                val id = idString.toInt

                // Parse the embedding string (a comma-separated list of doubles) into an array of Double
                val embedding = embeddingString.split(",").map(_.trim.toDouble)

                // Add the (id, embedding) pair to the map
                embeddingMapBuilder(id) = embedding

              case _ => // Ignore lines that don't match the expected pattern
            }
          }
        }.recover {
          case e: Exception =>
            logger.error(s"An error occurred while reading the file ${file.getName}: ${e.getMessage}")
        }
      }

      logger.info("Embeddings read from file.")


      // Convert the mutable map to an immutable one
      embeddingMap = embeddingMapBuilder.toMap

      if (embeddingMap.nonEmpty) {
        // Set embeddingDim to the length of the first embedding vector
        embeddingDim = embeddingMap.head._2.length

        // Set oov vector to predefined random embedding vector
        Nd4j.getRandom.setSeed(12345L)
        oov = Nd4j.rand(1, embeddingDim).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)
      } else {
        logger.warn("WARNING: Embedding map empty, embeddingDim and OOV not set")
      }

      if (encodeType != "default") {
        encodingType = getEncodingType(encodeType)
        logger.info(s"Encoding type set to ${encodeType}. If invalid, will default to r50k_base")
      }

      logger.info("Embeddings map created")


    } else {
      logger.error(s"The provided path is not a directory: $directoryPath")
    }
  }

  def getVocabSize: Int = embeddingMap.size

  def getEmbeddingDim: Int = embeddingDim

  def getMap: Map[Int, Array[Double]] = embeddingMap

  // Create sliding windows for inputs and targets with positional embeddings
  def createSlidingWindowsWithPositionalEmbedding(tokenString: String, windowSize: Int): util.ArrayList[DataSet] = {

    val tokens: Array[Int] = encoding.encode(tokenString).toArray
    val dataSetList: util.ArrayList[DataSet] = new util.ArrayList[DataSet]()

    for (i <- 0 until tokens.length - windowSize) {

      // Extract the input window (windowSize tokens)
      val inputWindow = new Array[Int](windowSize)
      System.arraycopy(tokens, i, inputWindow, 0, windowSize)

      // Extract the target token (the token right after the window)
      val targetToken = tokens(i + windowSize)

      // Convert input tokens into embeddings
      val inputEmbeddings: INDArray = tokenizeAndEmbed(inputWindow)

      // Compute positional embeddings
      val positionalEmbeddings: INDArray = computePositionalEmbedding(windowSize)

      // Ensure neither inputEmbeddings nor positionalEmbeddings are empty before performing the addition
      if (inputEmbeddings.isEmpty || positionalEmbeddings.isEmpty) {
        throw new IllegalStateException("Cannot perform operation add on empty arrays.")
      }

      // Add positional embeddings to the word embeddings
      val positionAwareEmbedding: INDArray = inputEmbeddings.add(positionalEmbeddings)

      // Convert the target token into an embedding
      val targetEmbedding: INDArray = tokenizeAndEmbed(Array(targetToken))

      // Add to the dataset (input is the window with positional embeddings, target is the next word)
      dataSetList.add(new DataSet(positionAwareEmbedding, targetEmbedding))
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

      // Reshape the data to [batchSize, embeddingDim, windowSize] -> [n, f, t]
      val windowSize: Int = (batchDataSet.getFeatures.shape()(0) / batchSize).toInt // Get the window size out of there without having to input it (im so smart)
      val features = batchDataSet.getFeatures.reshape(batchSize, embeddingDim, windowSize)

      val labels = batchDataSet.getLabels.reshape(batchSize, embeddingDim, 1)
      val paddedLabels = padLabelsWithZeros(labels, batchSize, embeddingDim, windowSize)

      // Update the dataset with the reshaped features and labels
      batchDataSet.setFeatures(features)
      batchDataSet.setLabels(paddedLabels)

      batchedData.add(batchDataSet)
    }

    // Any remaining data is ignored (dropped) since we don't process short batches.

    batchedData // Return the batched data
  }

  private def padLabelsWithZeros(labels: INDArray, batchSize: Int, embeddingDim: Int, sequenceLength: Int): INDArray = {
    // Create a new label array filled with zeros: [batchSize, outputSize, sequenceLength]
    val paddedLabels = Nd4j.zeros(batchSize, embeddingDim, sequenceLength)

    // Assign the actual label values to the first time step (the last dimension)
    for (i <- 0 until batchSize) {
      for (j <- 0 until embeddingDim) {
        // Cast the indices to Long to resolve the ambiguity in getDouble
        paddedLabels.putScalar(Array(i, j, 0), labels.getDouble(i.toLong, j.toLong))
      }
    }
    paddedLabels
  }

  private def tokenizeAndEmbed(tokens: Array[Int]): INDArray = {

    // Fetch embeddings for each token
    val embeddings = tokens.map { tokenId =>
      val embedding = embeddingMap.get(tokenId) match {
        case Some(embeddingArray) =>
          Nd4j.create(embeddingArray)
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
    else
      Nd4j.vstack(embeddings: _*)  // Stack embeddings vertically to create a matrix [num_tokens x embeddingDim]
  }

  // Compute sinusoidal positional embeddings for a given window size
  private def computePositionalEmbedding(windowSize: Int) = {

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

//  def main(args: Array[String]): Unit = {
//    val conf = new SparkConf().setAppName("Sliding Window Dataset").setMaster("local[*]")
//    val sc = new JavaSparkContext(conf)
//    initEmbeddingMap("/home/dbrun3/Desktop/441/CS441_Fall2024/output/embeddings")
//
//    // Example input data (could be sentences, tokens, etc.)
//    val sentences: Array[String] = Array("The brave man or the brave woman is one who looks life in the eye", "The pyramids therefore were tombs of the kings who built them while they were alive to be monuments to themselves when they were dead.")
//    val windowSize: Int = 5
//
//    // Parallelize the input data (convert array to an RDD)
//    val sentenceRDD: JavaRDD[String] = sc.parallelize(sentences.toSeq.asJava)
//
//    // Apply the sliding window logic to create the dataset
//    val slidingWindowDataset: JavaRDD[DataSet] = sentenceRDD.flatMap(sentence => {
//      createSlidingWindowsWithPositionalEmbedding(sentence, windowSize).iterator
//    })
//
//    // Collect and print the results (for demonstration)
//    slidingWindowDataset.collect.forEach(window => {
//
//      val inputEmbedding = window.getFeatures  // Input: Multiple tokens
//      val targetEmbedding = window.getLabels   // Target: Single token
//
//      // Print the shapes to confirm dimensions
//      System.out.println("Input shape: " + inputEmbedding.shapeInfoToString())  // Expect something like [4, 128]
//      System.out.println("Target shape: " + targetEmbedding.shapeInfoToString())  // Expect something like [1, 128]
//
//      // Optionally, print a subset or the full embedding
//      System.out.println("Input (first token's embedding): " + inputEmbedding.getRow(0).toString())
//      System.out.println("Target embedding: " + targetEmbedding.toString())
//
//    })
//
//    // Stop the Spark context
//    sc.stop()
//  }
}

