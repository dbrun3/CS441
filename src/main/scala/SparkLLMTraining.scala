import org.apache.spark.SparkConf
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.nd4j.linalg.dataset.DataSet
import org.slf4j.LoggerFactory
import com.typesafe.config.Config
import org.apache.spark.rdd.RDD
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.indexing.NDArrayIndex

import java.time.Instant
import java.util

object SparkLLMTraining {

  // Create a logger
  private val logger = LoggerFactory.getLogger(SparkLLMTraining.getClass)

  private def createRDDFromStrings(data: util.ArrayList[String], sc: JavaSparkContext): JavaRDD[String] = {
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

    if (args.length != 4) {
      logger.error("SparkLLMTraining usage: <input> <output> <embedding> <config>")
      return
    }

    // Set corpus input path
    val inputPath: String = args(0)

    // Set output file
    val modelPath: String = args(1)

    // Set embeddings from hw1 path
    val embeddingPath: String = args(2)

    // Set config file
    val confPath: String = args(3)

    // Initialize Spark context
    val sc: JavaSparkContext = createSparkContext("local[*]") // TODO: CHANGE BEFORE ASSEMBLY

    // Load config from file
    val config: Config = FileUtil.loadConfig(sc, confPath)

    // Spark config
    val averagingFrequency: Int = config.getInt("spark.averagingFrequency")
    val workerPrefetchNumBatches: Int = config.getInt("spark.workerPrefetchNumBatches")

    // Load embeddings and init embedding map using the spark context to access file in case of HDFS
    SlidingWindowWithPositionalEmbedding.initEmbeddingMap(sc, embeddingPath)

    // Embedding map to reference later
    val embeddingMap = SlidingWindowWithPositionalEmbedding.getEmbeddingMap
    val embeddingMapBroadcast = sc.broadcast(embeddingMap)

    // Model config (using embedding size from loaded file
    val embeddingDim = SlidingWindowWithPositionalEmbedding.getEmbeddingDim // Get embedding dimensions from loaded file
    val vocabSize = SlidingWindowWithPositionalEmbedding.getVocabSize
    val hiddenSize: Int = config.getInt("model.hiddenSize")
    val learningRate: Double = config.getDouble("model.learnRate")
    val batchSize: Int = config.getInt("model.batchSize")
    val windowSize: Int = config.getInt("model.windowSize")
    val numEpochs: Int = config.getInt("model.numEpochs")

    // Create RDD for sentences to parallelize window building
    val sentencesRDD: RDD[String] = FileUtil.getFileContentAsList(sc, inputPath)

    val totalSentences = sentencesRDD.count()
    println(s"$totalSentences sentences loaded. Starting sentence processing...")
    logger.info(s"$totalSentences sentences loaded. Starting sentence processing...")

    // Create sliding windows for each sentence in `sentencesRDD`, keeping everything distributed
    val slidingWindowsRDD: RDD[DataSet] = sentencesRDD.flatMap(sentence => {
      SlidingWindowWithPositionalEmbedding.createSlidingWindowsWithPositionalEmbedding(sentence, windowSize, vocabSize, embeddingDim, embeddingMapBroadcast)
    }).cache()


    val totalWindows = slidingWindowsRDD.count()
    println(s"$totalWindows windows created. Starting window batching...")
    logger.info(s"$totalWindows windows created. Starting window batching...")

    // Create batches windows distributed by partition
    val batchedWindowsRDD: RDD[DataSet] = slidingWindowsRDD.mapPartitions { iter =>
      SlidingWindowWithPositionalEmbedding.batchSlidingWindows(iter, batchSize, embeddingDim, vocabSize)
    }.cache()

    val batches = batchedWindowsRDD.count()
    logger.info(s"Number of batches created: $batches")
    println(s"Number of batches created: $batches")

    // Initialize model with embedding dimensions from hw1 and set num of neurons
    val model: MultiLayerNetwork = LLMModel.createModel(embeddingDim, hiddenSize, vocabSize, learningRate)

    // Set up the TrainingMaster configuration
    val trainingMaster: ParameterAveragingTrainingMaster = new ParameterAveragingTrainingMaster.Builder(32)
      .batchSizePerWorker(batchSize)
      .averagingFrequency(averagingFrequency)
      .workerPrefetchNumBatches(workerPrefetchNumBatches)
      .build()

    // Create a SparkDl4jMultiLayer with the Spark context and model
    val sparkModel: SparkDl4jMultiLayer = new SparkDl4jMultiLayer(sc, model, trainingMaster)

    // Set up listeners to collect stats during training
    model.setListeners(new ScoreIterationListener(10))

    // Train the model on the distributed RDD dataset
    logger.info("Starting training...")
    println("Starting training...")
    for (epoch <- 1 to numEpochs) {
      // Record the start time of the epoch
      val epochStartTime = Instant.now()

      logger.info(s"Epoch $epoch started")
      println(s"Epoch $epoch started")

      // Train for each epoch
      sparkModel.fit(batchedWindowsRDD)

      val score = sparkModel.getScore
      logger.info(s" Score: $score")
      println(s" Score: $score")

      // Record the end time and calculate the duration
      val epochEndTime = Instant.now()
      val epochDuration = java.time.Duration.between(epochStartTime, epochEndTime).toSeconds

      logger.info(s"Epoch $epoch finished in $epochDuration seconds")
      println(s"Epoch $epoch finished in $epochDuration seconds")
    }
    logger.info(s"Finished training.")

    // Save the model after training
    FileUtil.saveModel(sc, modelPath, sparkModel)
    logger.info(s"Model saved.")

    // Test on a test batch
    val firstXBatches = batchedWindowsRDD.take(5) // Take 5 test batches
    var correct = 0.0;
    var total = 0.0;

    // Iterate over each batch and each sequence within the batch
    for (batch <- firstXBatches) {
      val features = batch.getFeatures  // Shape: [batchSize, embeddingDim, sequenceLength]
      val labels = batch.getLabels      // Shape: [batchSize, vocabSize, sequenceLength]

      // Run the model to get predictions for the features
      val predictions = model.output(features)  // Predictions shape: [batchSize, vocabSize, sequenceLength]

      for (i <- 0 until batchSize) {
        // Get the last timestep for this sequence in the batch
        val labelLastTimestep = labels.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(windowSize - 1))
        val predictionLastTimestep = predictions.get(NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(windowSize - 1))

        // Find the index of the max value in the label for accuracy checking
        val labelMaxIndex = labelLastTimestep.argMax().getInt(0)

        // Flatten the predictions for easy sorting and get the top 5 indices and scores
        val flatPredictions = predictionLastTimestep.toDoubleVector
        val top5IndicesWithScores = flatPredictions.zipWithIndex
          .sortBy(-_._1)  // Sort by probability in descending order
          .take(5)        // Take the top 5
        val top5Indices = top5IndicesWithScores.map(_._2)
        val top5Scores = top5IndicesWithScores.map(_._1)

        // Check if the highest prediction matches the label
        val predictionMaxIndex = top5Indices.head
        if (labelMaxIndex == predictionMaxIndex) correct += 1
        total += 1

        // Display the top 5 predictions with translations
        println(s"Sequence $i - Label max index: $labelMaxIndex, Top 5 Prediction indices: ${top5Indices.mkString(", ")}")
        println(s"Sequence $i - Label (translated): ${SlidingWindowWithPositionalEmbedding.translateIndex(labelMaxIndex)}")
        top5Indices.zip(top5Scores).foreach { case (index, score) =>
          println(s"Prediction index: $index (translated: ${SlidingWindowWithPositionalEmbedding.translateIndex(index)}), Score: $score")
        }
      }
    }


    logger.info(f"Tests complete. Accuracy: ${(correct/total) * 100}%%")

    // Stop the Spark context after training
    sc.stop()
  }
}


