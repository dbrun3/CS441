import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.io.File
import scala.collection.mutable.ArrayBuffer
import scala.jdk.CollectionConverters.bufferAsJavaListConverter

class HW2Test extends AnyFlatSpec with Matchers {

  "computePositionalEmbedding" should "return an INDArray with the correct shape and values" in {
    val windowSize = 4
    val embeddingDim = 6

    // Call the function to compute the positional embeddings
    val embedding: INDArray = SlidingWindowWithPositionalEmbedding.computePositionalEmbedding(windowSize, embeddingDim)

    // Check if the embedding has the expected shape
    embedding.shape() should equal(Array(windowSize, embeddingDim))

    // Basic value checks for specific positions
    for (pos <- 0 until windowSize) {
      for (i <- 0 until embeddingDim by 2) {
        val expectedSin = Math.sin(pos / Math.pow(10000, 2.0 * i / embeddingDim))
        val expectedCos = Math.cos(pos / Math.pow(10000, 2.0 * i / embeddingDim))

        embedding.getDouble(pos.toLong, i.toLong) should be(expectedSin +- 1e-5)
        embedding.getDouble(pos.toLong, (i + 1).toLong) should be(expectedCos +- 1e-5)
      }
    }
  }

  "tokenizeAndEmbed" should "return an INDArray with the correct shape [tokens, embeddingDim]" in {
    val sc = SparkLLMTraining.createSparkContext("local[*]")
    SlidingWindowWithPositionalEmbedding.initEmbeddingMap(sc, "/home/dbrun3/Desktop/441/CS441_Fall2024/output/embeddings") // replace with your path
    val embeddingMap = SlidingWindowWithPositionalEmbedding.getEmbeddingMap
    val tokens = Array(1, 2, 3)
    val embeddingDim = 100

    // Tokenize and embed the input tokens
    val result: INDArray = SlidingWindowWithPositionalEmbedding.tokenizeAndEmbed(tokens, embeddingMap, embeddingDim)

    // Check if the result has the correct shape
    result.shape() should equal(Array(tokens.length, embeddingDim))

    // Optionally, perform basic checks on the values (e.g., ensure they are non-zero if embeddings exist for each token)
    tokens.zipWithIndex.foreach { case (token, index) =>
      val tokenEmbedding = result.getRow(index)
      tokenEmbedding.sumNumber().doubleValue() should not be 0.0
    }

    sc.stop()
  }

  "translateIndex" should "return '.' for index 13" in {
    val result = SlidingWindowWithPositionalEmbedding.translateIndex(13)
    result should equal(".")
  }

  "The batching DataSet transformation" should "reshape features and labels to the correct dimensions" in {
    // Define parameters
    val batchSize = 4
    val embeddingDim = 100
    val vocabSize = 50
    val sequenceLength = 10

    // Create a batch buffer with individual DataSets
    val batchBuffer = ArrayBuffer[DataSet]()

    // Add DataSets to the batch buffer to simulate the transformation process
    for (_ <- 0 until batchSize) {
      val features = Nd4j.rand(sequenceLength, embeddingDim)
      val labels = Nd4j.rand(sequenceLength, vocabSize)
      batchBuffer += new DataSet(features, labels)
    }

    // Merge the batch buffer and apply the transformation
    val batchDataSet = DataSet.merge(batchBuffer.asJava)
    val originalShape = batchDataSet.getFeatures.shape()
    val windowSize: Int = (originalShape(0) / batchSize).toInt

    val reshapedFeatures = batchDataSet.getFeatures.reshape(windowSize, batchSize, embeddingDim).permute(1, 2, 0)
    val reshapedLabels = batchDataSet.getLabels.reshape(batchSize, vocabSize, windowSize)

    batchDataSet.setFeatures(reshapedFeatures)
    batchDataSet.setLabels(reshapedLabels)

    // Check dimensions of reshaped features and labels
    val reshapedFeatureShape = batchDataSet.getFeatures.shape()
    val reshapedLabelShape = batchDataSet.getLabels.shape()

    reshapedFeatureShape should equal(Array(batchSize, embeddingDim, sequenceLength))
    reshapedLabelShape should equal(Array(batchSize, vocabSize, sequenceLength))
  }




  // Helper method to delete all files in a directory
  def clearDirectory(directoryPath: String): Unit = {
    val directory = new File(directoryPath)
    if (directory.exists && directory.isDirectory) {
      directory.listFiles().foreach(_.delete())
    }
  }

  "SparkLLMTraining" should "create a model.zip upon success" in {

    // Replace with your paths before testing obv
    val args: Array[String] = Array(
      "/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/input",
      "/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/model/LLM_Spark_Model.zip",
      "/home/dbrun3/Desktop/441/CS441_Fall2024/output/embeddings",
      "/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/test.conf"
    )
    val outputPath = "/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/model/" // Replace with your directory path

    // Step 1: Clear the output directory
    clearDirectory(outputPath)
    val directory = new File(outputPath)

    val filesBeforeRun = directory.listFiles()
    filesBeforeRun shouldBe empty

    // Step 2: Run training
    SparkLLMTraining.main(args)

    // Step 3: Check if any files were created in the directory
    val filesAfterRun = directory.listFiles()

    // Clear directory afterward
    clearDirectory(outputPath)

    // Assert that there is at least one file in the directory
    filesAfterRun should not be empty
  }
}
