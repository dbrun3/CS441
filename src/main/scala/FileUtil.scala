import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.slf4j.LoggerFactory

import scala.io.Source
import scala.util.Using
import java.io.{File, FileOutputStream, IOException, InputStream, OutputStream}
import scala.collection.mutable
import scala.io.Source
import scala.util.Using
import java.util
import scala.collection.convert.ImplicitConversions.`iterator asJava`
import scala.jdk.CollectionConverters._
import org.apache.spark.SparkContext
import com.typesafe.config.{Config, ConfigFactory}
import org.deeplearning4j.util.ModelSerializer
import org.apache.hadoop.fs.{FileSystem, Path}

import java.net.URI
import java.nio.file.{Files, Paths}
import org.apache.spark.SparkContext
import org.apache.spark.api.java.JavaRDD
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

object FileUtil {

  private val logger = LoggerFactory.getLogger(SparkLLMTraining.getClass)

  def getFileContentAsList(sc: SparkContext, directoryPath: String): JavaRDD[String] = {

    // Use SparkContext textFile to read all files in the directory from S3
    try {
      sc.textFile(directoryPath + "/*")
    } catch {
      case e: Exception =>
        logger.error(s"An error occurred while reading the S3 path: $directoryPath", e)
        sc.emptyRDD[String]
    }
  }


  def loadEmbeddings(sc: SparkContext, directoryPath: String): Map[Int, INDArray] = {

    // Read all lines across files in the directory
    val lines = sc.textFile(directoryPath + "/*")

    // Process each partition in a distributed manner
    val embeddingPairs = lines.mapPartitions { partition =>
      val partitionMap = mutable.Map[Int, INDArray]()

      partition.foreach { line =>
        parseLineContent(line, partitionMap) // Populate partition-specific map
      }

      partitionMap.iterator // Return an iterator of (key, value) pairs
    }

    // Reduce to combine all pairs into a single map
    embeddingPairs.collectAsMap().toMap
  }

  private def parseLineContent(line: String, embeddingMapBuilder: mutable.Map[Int, INDArray]): Unit = {
    // Regex to extract the ID and the embedding vector
    val pattern = """word:\s+\S+\s+id:\s+(\d+)\s+freq:\s+\d+\s+\[([^\]]+)\]""".r

    line match {
      case pattern(idString, embeddingString) =>
        val id = idString.toInt
        val embedding = embeddingString.split(",").map(_.trim.toDouble)

        val embeddingINDArray = Nd4j.create(embedding)          // Convert Array[Double] to INDArray
        val normalizedINDArray = Transforms.unitVec(embeddingINDArray)  // normalize
        embeddingMapBuilder(id) = normalizedINDArray

      case _ => // Ignore lines that don't match the expected pattern
    }
  }

  // Load configuration based on path (either local or S3)
  def loadConfig(sc: SparkContext, confPath: String): Config = {
    val configData = sc.textFile(confPath).collect().mkString("\n")
    ConfigFactory.parseString(configData)
  }

  // Save model either locally or on S3
  @throws[IOException]
  def saveModel(sc: SparkContext, modelPath: String, sparkModel: SparkDl4jMultiLayer): Unit = {
    if (modelPath.startsWith("s3://")) {
      // Use Hadoop FileSystem to save the model to S3
      val hadoopConf = sc.hadoopConfiguration
      val fs = FileSystem.get(new URI(modelPath), hadoopConf)
      val outputStream: OutputStream = fs.create(new Path(modelPath))
      ModelSerializer.writeModel(sparkModel.getNetwork, outputStream, true)
      outputStream.close()
    } else {
      // Save locally using java.io.File
      ModelSerializer.writeModel(sparkModel.getNetwork, new File(modelPath), true)
    }
  }

  // Method to load the pretrained model from either local or S3
  @throws[IOException]
  def loadPretrainedModel(sc: SparkContext, modelPath: String): MultiLayerNetwork = {
    if (modelPath.startsWith("s3://")) {
      // Use Hadoop FileSystem to load the model from S3
      val hadoopConf = sc.hadoopConfiguration
      val fs = FileSystem.get(new URI(modelPath), hadoopConf)
      val inputStream: InputStream = fs.open(new Path(modelPath))
      val model = ModelSerializer.restoreMultiLayerNetwork(inputStream)
      inputStream.close()
      model
    } else {
      // Load locally using java.io.File
      val file = new File(modelPath)
      ModelSerializer.restoreMultiLayerNetwork(file)
    }
  }




}
