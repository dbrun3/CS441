import org.bytedeco.hdf5.DataSet
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import java.util


object SlidingWindowWithPositionalEmbedding {
  // Create sliding windows for inputs and targets with positional embeddings
  def createSlidingWindowsWithPositionalEmbedding(tokens: Array[String], windowSize: Int): util.List[DataSet] = {
    val dataSetList = new util.ArrayList[DataSet]()
    for (i <- 0 to tokens.length - windowSize) {
      // Extract the input window (windowSize tokens)
      val inputWindow = new Array[String](windowSize)
      System.arraycopy(tokens, i, inputWindow, 0, windowSize)
      // Extract the target token (the token right after the window)
      val targetToken = tokens(i + windowSize)
      // Convert input tokens into embeddings
      val inputEmbeddings = tokenizeAndEmbed(inputWindow) // Embedding for words

      // Add positional embeddings to the word embeddings
      val positionalEmbeddings = computePositionalEmbedding(windowSize)
      val positionAwareEmbedding = inputEmbeddings.add(positionalEmbeddings)
      // Convert the target token into an embedding
      val targetEmbedding = tokenizeAndEmbed(Array[String](targetToken))
      // Add to the dataset (input is the window with positional embeddings, target is the next word)
      dataSetList.add(new Nothing(positionAwareEmbedding, targetEmbedding))
    }
    dataSetList
  }

  // Dummy method to simulate tokenization and embedding (replace with actual embedding code)
  private def tokenizeAndEmbed(tokens: Array[String]) = {
    // For simplicity, let's assume each word is embedded as a 1x128 vector
    Nd4j.rand(tokens.length, 128) // Generate random embeddings

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
    // Example sentence (tokenized)
    val sentence = Array("The", "cat", "sat", "on", "a", "mat")
    // Create sliding windows of size 4 with positional embeddings
    val windowSize = 4
    val slidingWindows = createSlidingWindowsWithPositionalEmbedding(sentence, windowSize)
    // Output the number of sliding windows created
    System.out.println("Number of sliding windows with positional embeddings: " + slidingWindows.size)
  }
}