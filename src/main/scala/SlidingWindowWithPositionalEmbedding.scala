import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.DataSet
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.SparkConf

import java.util

object SlidingWindowWithPositionalEmbedding {
  // Create sliding windows for inputs and targets with positional embeddings
  def createSlidingWindowsWithPositionalEmbedding(tokenString: String, windowSize: Int): util.ArrayList[DataSet] = {

    val tokens: Array[String] = tokenString.split(" ") //TODO: Replace with JTokkit tokenizatation
    val dataSetList: util.ArrayList[DataSet] = new util.ArrayList[DataSet]()

    for (i <- 0 to tokens.length - windowSize) {
      // Extract the input window (windowSize tokens)
      val inputWindow = new Array[String](windowSize)
      System.arraycopy(tokens, i, inputWindow, 0, windowSize)

      // Extract the target token (the token right after the window)
      val targetToken = tokens(i + windowSize)

      // Convert input tokens into embeddings
      val inputEmbeddings: INDArray = tokenizeAndEmbed(inputWindow)

      // Add positional embeddings to the word embeddings
      val positionalEmbeddings: INDArray = computePositionalEmbedding(windowSize)
      val positionAwareEmbedding: INDArray = inputEmbeddings.add(positionalEmbeddings)

      // Convert the target token into an embedding
      val targetEmbedding: INDArray = tokenizeAndEmbed(Array(targetToken))

      // Add to the dataset (input is the window with positional embeddings, target is the next word)
      dataSetList.add(new DataSet(positionAwareEmbedding, targetEmbedding))
    }

    dataSetList
  }

  // Dummy method to simulate tokenization and embedding (replace with actual embedding code)
  private def tokenizeAndEmbed(tokens: Array[String]) = {
    // For simplicity, let's assume each word is embedded as a 1x128 vector
    Nd4j.rand(tokens.length, 128) // Generate random embeddings //TODO: Replace with embeddings loaded from HW1

  }

  // Compute sinusoidal positional embeddings for a given window size
  private def computePositionalEmbedding(windowSize: Int) = {
    val embeddingDim = 128 // Dimensionality of word embeddings

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

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("Sliding Window Dataset").setMaster("local[*]")
    val sc = new JavaSparkContext(conf)

    // Example input data (could be sentences, tokens, etc.)
    val sentences: Array[String] = Array("The quick brown fox jumps over the lazy dog", "This is another sentence for testing sliding windows")
    val windowSize: Int = 5

    // Parallelize the input data (convert array to an RDD)
    val sentenceRDD: JavaRDD[String] = sc.parallelize(util.Arrays.asList(sentences))

    // Apply the sliding window logic to create the dataset
    val slidingWindowDataset: JavaRDD[DataSet] = sentenceRDD.flatMap(sentence => {
      SlidingWindowWithPositionalEmbedding.createSlidingWindowsWithPositionalEmbedding(sentence, windowSize).iterator
    })

    // Collect and print the results (for demonstration)
    slidingWindowDataset.collect.forEach((window) => {

      val inputEmbedding = window.getFeatures  // Input: Multiple tokens
      val targetEmbedding = window.getLabels   // Target: Single token

      // Print the shapes to confirm dimensions
      System.out.println("Input shape: " + inputEmbedding.shapeInfoToString())  // Expect something like [4, 128]
      System.out.println("Target shape: " + targetEmbedding.shapeInfoToString())  // Expect something like [1, 128]

      // Optionally, print a subset or the full embedding
      System.out.println("Input (first token's embedding): " + inputEmbedding.getRow(0).toString)
      System.out.println("Target embedding: " + targetEmbedding.toString)

    })

    // Stop the Spark context
    sc.stop()
  }
}

