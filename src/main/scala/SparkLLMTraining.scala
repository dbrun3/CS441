import org.apache.spark.SparkConf
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.nd4j.linalg.dataset.DataSet
import org.deeplearning4j.util.ModelSerializer

import java.io.File
import java.util

object SparkLLMTraining {

  def createRDDFromData(data: util.List[DataSet], sc: JavaSparkContext): JavaRDD[DataSet] = {
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
    // Initialize Spark context
    val sc: JavaSparkContext = createSparkContext

    // Create your LLM model using DL4J
    val model: MultiLayerNetwork = null //TODO: Create a model

    // Prepare data (you can use the sliding window data from the previous step)
    // Example input data (could be sentences, tokens, etc.)
    val sentences: Array[String] = Array("The quick brown fox jumps over the lazy dog", "This is another sentence for testing sliding windows")
    val windowSize: Int = 5

    // Parallelize the input data (convert array to an RDD)
    val sentenceRDD: JavaRDD[String] = sc.parallelize(util.Arrays.asList(sentences))

    // Apply the sliding window logic to create the dataset
    val slidingWindowDataset: JavaRDD[DataSet] = sentenceRDD.flatMap(sentence => {
      SlidingWindowWithPositionalEmbedding.createSlidingWindowsWithPositionalEmbedding(sentence, windowSize).iterator
    })

    // Set up the TrainingMaster configuration
    val trainingMaster: ParameterAveragingTrainingMaster = new ParameterAveragingTrainingMaster.Builder(32)
      .batchSizePerWorker(32)
      .averagingFrequency(5)
      .workerPrefetchNumBatches(2)
      .build()// Batch size on each Spark worker.averagingFrequency(5)// Frequency of parameter averaging.workerPrefetchNumBatches(2).build

    // Create a SparkDl4jMultiLayer with the Spark context and model
    val sparkModel = new SparkDl4jMultiLayer(sc, model, trainingMaster)

    // Set listeners to monitor the training progress
    model.setListeners(new ScoreIterationListener()(10))

    // Train the model on the distributed RDD dataset
    sparkModel.fit(slidingWindowDataset)

    // Save the model after training
    ModelSerializer.writeModel(sparkModel.getNetwork, new File("LLM_Spark_Model.zip"), true) //TODO: For now use java.io but eventually add s3 switch

    // Stop the Spark context after training
    sc.stop()
  }
}


