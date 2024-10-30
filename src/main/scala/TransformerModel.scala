import SparkLLMTraining.createSparkContext
import org.apache.spark.api.java.JavaSparkContext
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms

import java.io.IOException


object TransformerModel {

  def getLastWindowWithPadding(contextTokenized: Array[Int], windowSize: Int): Array[Int] = {
    // If the token array is shorter than windowSize, pad with zeros at the beginning
    if (contextTokenized.length < windowSize) {
      // Calculate the number of padding zeros needed
      val paddingSize = windowSize - contextTokenized.length
      // Create a new array with padding zeros followed by the tokens
      Array.fill(paddingSize)(0) ++ contextTokenized
    } else {
      // Otherwise, return the last windowSize tokens
      contextTokenized.takeRight(windowSize)
    }
  }

  // Method to generate the next word based on the query using the pretrained model
  def generateNextWord(contextEmbedding: INDArray, model: MultiLayerNetwork): Int = {
    // Tokenize context and convert to embedding (tokenization + embedding is done as part of homework 1)
    val reshapedContextEmbedding: INDArray = contextEmbedding.reshape(1, contextEmbedding.columns(), contextEmbedding.rows())  // [1, embedding_dim, window_size]
    val rawOutput: INDArray = model.output(reshapedContextEmbedding)  // [1, embedding_dim, window_size]
    val output = rawOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(4 - 1))
    val index = output.argMax().getInt(0)
    index
  }

  // Method to generate a full sentence based on the seed text
  def generateSentence(seedText: String, model: MultiLayerNetwork, maxWords: Int, windowSize: Int): String = {
    var generatedText: String = seedText
    var finished = false
    var i = 0

    // Tokenize context and convert to embedding (2D)
    var contextTokenized: Array[Int] = SlidingWindowWithPositionalEmbedding.encodeTokens(seedText)

    while (i < maxWords && !finished) {

      val tokenWindow: Array[Int] = getLastWindowWithPadding(contextTokenized, windowSize)
      val contextEmbeddings: INDArray = SlidingWindowWithPositionalEmbedding.tokenizeAndEmbed(tokenWindow)
      val index: Int = generateNextWord(contextEmbeddings, model)

      contextTokenized = (contextTokenized :+ index).takeRight(windowSize)
      val nextWord: String = SlidingWindowWithPositionalEmbedding.translateIndex(index)
      if(nextWord == ".") finished = true

      generatedText += nextWord

      println(generatedText)

      i += 1
    }
    generatedText
  }

//
//  @throws[IOException]
//  def main(args: Array[String]): Unit = {
//
//    val sc: JavaSparkContext = createSparkContext("local[*]") // TODO: CHANGE BEFORE ASSEMBLY
//    // Load the pretrained transformer model from file
//    val modelPath = "/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/model/model.zip" // Path to the pretrained model file
//    val embeddingPath = "/home/dbrun3/Desktop/441/CS441_Fall2024/output/embeddings"
//
//    SlidingWindowWithPositionalEmbedding.initEmbeddingMap(sc, embeddingPath)
//
//    println("mapsize: " + SlidingWindowWithPositionalEmbedding.getMap.size)
//
//    val model = FileUtil.loadPretrainedModel(sc, modelPath)
//    // Generate text using the pretrained model
//    val query = "is the most valuable"
//    val generatedSentence = generateSentence(query, model, 5, 5) // Generate a sentence with max 5 words
//
//    System.out.println("Generated Sentence: " + generatedSentence)
//  }
}