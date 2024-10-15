import org.apache.spark.SparkConf
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.nd4j.linalg.dataset.DataSet
import org.deeplearning4j.util.ModelSerializer
import org.deeplearning4j.ui.model.stats.StatsListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage
import org.slf4j.LoggerFactory
import com.typesafe.config.{Config, ConfigFactory}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork

import java.io.File
import java.util

object SparkLLMTraining {

  // Create a logger
  private val logger = LoggerFactory.getLogger(SparkLLMTraining.getClass)

  def createRDDFromStrings(data: util.ArrayList[String], sc: JavaSparkContext): JavaRDD[String] = {
    // Parallelize your data into a distributed RDD
    val rddData = sc.parallelize(data)
    rddData
  }

  def createRDDFromData(data: util.ArrayList[DataSet], sc: JavaSparkContext): JavaRDD[DataSet] = {
    // Parallelize your data into a distributed RDD
    val rddData = sc.parallelize(data)
    rddData
  }

  def createSparkContext(environment: String): JavaSparkContext = {
    // Configure Spark for local or cluster mode
    val sparkConf = new SparkConf()
      .setAppName("DL4J-LanguageModel-Spark")
      .setMaster(environment) // "local[*]" for local testing, or use "yarn", "mesos", etc. in a cluster

    // Create Spark context
    val sc: JavaSparkContext = new JavaSparkContext(sparkConf)
    sc
  }


  def main(args: Array[String]): Unit = {

    //TODO get file paths from args and figure out how to adapt to AWS like hw1 (but without hdfs)
    // Set input directory
    val inputPath: String = "/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/input"

    // Set output file
    val modelPath: String = "/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/model/LLM_Spark_Model.zip"

    // Set config file
    val confPath: String = "/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/application.conf"

    // Set embeddings from hw1 path
    val embeddingPath: String = "/home/dbrun3/Desktop/441/CS441_Fall2024/output/embeddings"

    // Initialize Spark context
    val sc: JavaSparkContext = createSparkContext("local[*]") // TODO: CHANGE BEFORE ASSEMBLY

    // Load config from file
    val config: Config = FileUtil.loadConfig(sc, confPath)

    // Spark config
    val averagingFrequency: Int = config.getInt("spark.averagingFrequency")
    val workerPrefetchNumBatches: Int = config.getInt("spark.workerPrefetchNumBatches")

    // Load embeddings and init embedding map using the spark context to access file in case of HDFS
    SlidingWindowWithPositionalEmbedding.initEmbeddingMap(sc, embeddingPath)

    // Model config (using embedding size from loaded file
    val embeddingDim = SlidingWindowWithPositionalEmbedding.getEmbeddingDim // Get embedding dimensions from loaded file
    val hiddenSize: Int = config.getInt("model.hiddenSize")
    val batchSize: Int = config.getInt("model.batchSize")
    val windowSize: Int = config.getInt("model.windowSize")
    val numEpochs: Int = config.getInt("model.numEpochs")

    // Example input data (could be sentences, tokens, etc.) //TODO: Read sentences (lines) from each file in an input folder
    val sentence: String = "The Trojans thought this was a sign from the gods, or an omen as they would have said, that they should not believe Laocoon; so they determined to take the horse into the city against his advice. The horse was so big, however, that it would not go through the gates, and in order to get it inside of the walls they had to tear down part of the wall itself. When night fell, the Greek soldiers came out of the horse and opened the gates of the city. The other Greeks, who had been waiting just out of sight, returned and entered through the gates and the hole the Trojans had made in the wall. Troy was easily conquered then, and the city was burned to the ground, and Helen’s husband carried her back to Greece. For reason of this horse trick, we still have a saying, “Beware of the Greeks bearing gifts,” which is as much as to say, “Look out for an enemy who makes you a present.”"

    // Load input from directory using the spark context to access file in case of HDFS
    val sentences: util.ArrayList[String] = FileUtil.getFileContentAsList(sc, inputPath)

    // Initialize model with embedding dimensions from hw1 and set num of neurons
    val model: MultiLayerNetwork = LLMModel.createModel(embeddingDim, hiddenSize)

    // Create windows, batch them and convert to RDD
    val windows: util.ArrayList[DataSet] = SlidingWindowWithPositionalEmbedding.createSlidingWindowsWithPositionalEmbedding(sentences.get(0), windowSize)
    logger.info(s"Sliding windows created. Total number of windows: ${windows.size()}")
    val batchedWindows: util.ArrayList[DataSet] = SlidingWindowWithPositionalEmbedding.batchSlidingWindows(windows, batchSize)
    logger.info(s"Batched windows created. Total number of batches: ${batchedWindows.size()}")
    val batchedRDD: JavaRDD[DataSet] = createRDDFromData(batchedWindows, sc)
    logger.info(s"Batched RDD created. Total partitions in RDD: ${batchedRDD.getNumPartitions}")

    // Set up the TrainingMaster configuration
    val trainingMaster: ParameterAveragingTrainingMaster = new ParameterAveragingTrainingMaster.Builder(32)
      .batchSizePerWorker(batchSize)
      .averagingFrequency(averagingFrequency)
      .workerPrefetchNumBatches(workerPrefetchNumBatches)
      .build()

    // Create a SparkDl4jMultiLayer with the Spark context and model
    val sparkModel: SparkDl4jMultiLayer = new SparkDl4jMultiLayer(sc, model, trainingMaster)

    // Set up the UI server
    val uiServer = UIServer.getInstance()

    // Create FileStatsStorage instance to store training stats
    val statsStorage = new InMemoryStatsStorage()

    // Attach the StatsStorage instance to the UI server
    uiServer.attach(statsStorage)

    // Set up listeners to collect stats during training
    model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10))

    // Train the model on the distributed RDD dataset
    logger.info("Starting training...")
    for (epoch <- 1 to numEpochs) {
      logger.info(s"Epoch $epoch started")
      sparkModel.fit(batchedRDD) // Train for each epoch
      val score = sparkModel.getScore
      logger.info(s" Score: $score")
      logger.info(s"Epoch $epoch finished")
    }
    logger.info("Finished training.")

    // Save the model after training
    FileUtil.saveModel(sc, modelPath, sparkModel)

    // Stop the Spark context after training
    sc.stop()
  }
}


