import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingRegistry
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.DataSet
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.SparkConf

import java.io.File
import java.util
import scala.io.Source
import scala.collection.mutable
import scala.jdk.CollectionConverters.seqAsJavaListConverter
import scala.util.Using

object SlidingWindowWithPositionalEmbedding {

  // Map of token ids to their vector embeddings + vector dimensions
  private var embeddingMap: Map[Int, Array[Double]] = Map() // vars since we are initializing it during runtime
  private var embeddingDim: Int = 0

  def initEmbeddingMap(directoryPath: String): Unit = {
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
            val pattern = """word:\s+\S+\s+id:\s+(\d+)\s+freq:\s+\d+\s+\[([^\]]+)\]""".r

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
            println(s"An error occurred while reading the file ${file.getName}: ${e.getMessage}")
        }
      }

      // Convert the mutable map to an immutable one
      embeddingMap = embeddingMapBuilder.toMap

      // Set embeddingDim to the length of the first embedding vector
      if (embeddingMap.nonEmpty) {
        embeddingDim = embeddingMap.head._2.length
      }
    } else {
      println(s"The provided path is not a directory: $directoryPath")
    }
  }

  // Create sliding windows for inputs and targets with positional embeddings
  def createSlidingWindowsWithPositionalEmbedding(tokenString: String, windowSize: Int): util.ArrayList[DataSet] = {

    val encodingType: EncodingType = getEncodingType("r50k_base")
    val registry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()
    val encoding: Encoding = registry.getEncoding(encodingType)

    val tokens: Array[Int] = encoding.encode(tokenString).toArray

    val dataSetList: util.ArrayList[DataSet] = new util.ArrayList[DataSet]()

    for (i <- 0 to tokens.length - windowSize - 1) {
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

  private def tokenizeAndEmbed(tokens: Array[Int]): INDArray = {

    // Fetch embeddings for each token
    val embeddings = tokens.map { tokenId =>
      val embedding = embeddingMap.get(tokenId) match {
        case Some(embeddingArray) =>
          Nd4j.create(embeddingArray)
        case None =>
          Nd4j.rand(1, embeddingDim).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)
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
      throw new IllegalStateException("No embeddings found for the given tokens.")
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
      case _             => throw new IllegalArgumentException(s"Unknown encoding type: $encoding")
    }
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Sliding Window Dataset").setMaster("local[*]")
    val sc = new JavaSparkContext(conf)
    initEmbeddingMap("/home/dbrun3/Desktop/441/CS441_Fall2024/output/embeddings")

    // Example input data (could be sentences, tokens, etc.)
    val sentences: Array[String] = Array("The brave man or the brave woman is one who looks life in the eye", "The pyramids therefore were tombs of the kings who built them while they were alive to be monuments to themselves when they were dead.")
    val windowSize: Int = 5

    // Parallelize the input data (convert array to an RDD)
    val sentenceRDD: JavaRDD[String] = sc.parallelize(sentences.toSeq.asJava)

    // Apply the sliding window logic to create the dataset
    val slidingWindowDataset: JavaRDD[DataSet] = sentenceRDD.flatMap(sentence => {
      createSlidingWindowsWithPositionalEmbedding(sentence, windowSize).iterator
    })

    // Collect and print the results (for demonstration)
    slidingWindowDataset.collect.forEach((window) => {

      val inputEmbedding = window.getFeatures  // Input: Multiple tokens
      val targetEmbedding = window.getLabels   // Target: Single token

      // Print the shapes to confirm dimensions
      System.out.println("Input shape: " + inputEmbedding.shapeInfoToString())  // Expect something like [4, 128]
      System.out.println("Target shape: " + targetEmbedding.shapeInfoToString())  // Expect something like [1, 128]

      // Optionally, print a subset or the full embedding
      System.out.println("Input (first token's embedding): " + inputEmbedding.getRow(0).toString())
      System.out.println("Target embedding: " + targetEmbedding.toString())

    })

    // Stop the Spark context
    sc.stop()
  }
}

