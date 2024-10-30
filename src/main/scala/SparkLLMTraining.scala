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
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms

import java.io.File
import java.util
import scala.collection.convert.ImplicitConversions.`iterable AsScalaIterable`

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

    // Model config (using embedding size from loaded file
    val embeddingDim = SlidingWindowWithPositionalEmbedding.getEmbeddingDim // Get embedding dimensions from loaded file
    val vocabSize = SlidingWindowWithPositionalEmbedding.getVocabSize
    val hiddenSize: Int = config.getInt("model.hiddenSize")
    val learningRate: Double = config.getDouble("model.learnRate")
    val batchSize: Int = config.getInt("model.batchSize")
    val windowSize: Int = config.getInt("model.windowSize")
    val numEpochs: Int = config.getInt("model.numEpochs")

    // Load input from directory using the spark context to access file in case of HDFS
    val sentences: util.ArrayList[String] = FileUtil.getFileContentAsList(sc, inputPath)

    // Create RDD for sentences to parallelize window building
    val sentencesRDD: JavaRDD[String] = createRDDFromStrings(sentences, sc)

    // Parallelize window creation using mapPartitions to keep the logic distributed
    val batchedWindowsRDD: JavaRDD[DataSet] = sentencesRDD.mapPartitions(sentenceIter => {
      val windowsList = new util.ArrayList[DataSet]()

      sentenceIter.forEachRemaining(sentence => {
        // Create windows and batch them
        val windows: util.ArrayList[DataSet] = SlidingWindowWithPositionalEmbedding.createSlidingWindowsWithPositionalEmbedding(sentence, windowSize)
        val batchedWindows: util.ArrayList[DataSet] = SlidingWindowWithPositionalEmbedding.batchSlidingWindows(windows, batchSize)

        // Add all batched windows to the list
        windowsList.addAll(batchedWindows)
      })

      // Return an iterator over the batched windows
      windowsList.iterator()
    })

    // Log resulting batch sizes
    batchedWindowsRDD.foreach(batch => logger.info(s"Batch processed with ${batch.size} elements"))

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
      sparkModel.fit(batchedWindowsRDD) // Train for each epoch
      val score = sparkModel.getScore
      logger.info(s" Score: $score")
      logger.info(s"Epoch $epoch finished")
    }
    val score = sparkModel.getScore
    logger.info(s"Finished training. Final score $score")

    // TODO Copy to test.scala later...
    // Extract the first x batches from the RDD
    val firstXBatches = batchedWindowsRDD.take(5) // 5 test batches
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

    // Save the model after training
    FileUtil.saveModel(sc, modelPath, sparkModel)

    // Stop the Spark context after training
    sc.stop()
  }
}


