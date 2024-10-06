import org.apache.spark.SparkConf
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.nd4j.linalg.dataset.DataSet
import org.deeplearning4j.util.ModelSerializer

import java.io.File
import java.util
import scala.jdk.CollectionConverters.seqAsJavaListConverter

object SparkLLMTraining {

  def createRDDFromData(data: util.ArrayList[DataSet], sc: JavaSparkContext): JavaRDD[DataSet] = {
    // Parallelize your data into a distributed RDD
    val rddData = sc.parallelize(data)
    rddData
  }

  def createSparkContext: JavaSparkContext = {
    // Configure Spark for local or cluster mode
    val sparkConf = new SparkConf()
      .setAppName("DL4J-LanguageModel-Spark")
      .setMaster("local[*]") // For local testing, or use "yarn", "mesos", etc. in a cluster

    // Create Spark context
    val sc: JavaSparkContext = new JavaSparkContext(sparkConf)
    sc
  }


  def main(args: Array[String]): Unit = {

    // Prepare data (you can use the sliding window data from the previous step)
    // Example input data (could be sentences, tokens, etc.)
    val sentence: String = "The brave man or the brave woman is one who looks life in the eye"
    val windowSize: Int = 5

    // Initialize sliding window with embedding data from HW1
    SlidingWindowWithPositionalEmbedding.initEmbeddingMap("/home/dbrun3/Desktop/441/CS441_Fall2024/output/embeddings")

    // Define the model configuration based on embedding data from HW1
    val embeddingDim = SlidingWindowWithPositionalEmbedding.getEmbeddingDim // Embedding dimensions
    val hiddenUnits = 256                                                   // hidden units (neurons)
    val numClasses = SlidingWindowWithPositionalEmbedding.getVocabSize      // vocabulary size

    // Initialize Spark context
    val sc: JavaSparkContext = createSparkContext

    val model = LLMModel.createModel(embeddingDim, hiddenUnits, numClasses)

    val windows: util.ArrayList[DataSet] = SlidingWindowWithPositionalEmbedding.createSlidingWindowsWithPositionalEmbedding(sentence, windowSize)
    val rddData: JavaRDD[DataSet] = createRDDFromData(windows, sc)

    // Set up the TrainingMaster configuration
    val trainingMaster: ParameterAveragingTrainingMaster = new ParameterAveragingTrainingMaster.Builder(32)
      .batchSizePerWorker(32)
      .averagingFrequency(5)
      .workerPrefetchNumBatches(2)
      .build()// Batch size on each Spark worker.averagingFrequency(5)// Frequency of parameter averaging.workerPrefetchNumBatches(2).build

    // Create a SparkDl4jMultiLayer with the Spark context and model
    val sparkModel = new SparkDl4jMultiLayer(sc, model, trainingMaster)

    // Train the model on the distributed RDD dataset
    sparkModel.fit(rddData)

    // Save the model after training
    ModelSerializer.writeModel(sparkModel.getNetwork, new File("/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/model/LLM_Spark_Model.zip"), true) //TODO: For now java.io is fine

    // Stop the Spark context after training
    sc.stop()
  }
}


