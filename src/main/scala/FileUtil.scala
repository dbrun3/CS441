import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.slf4j.LoggerFactory

import scala.io.Source
import scala.util.Using
import java.io.File
import java.util.ArrayList
import scala.collection.mutable
import scala.io.Source
import scala.util.Using
import java.io.File
import java.util
import scala.collection.convert.ImplicitConversions.`iterator asJava`
import scala.jdk.CollectionConverters._
import java.io.{File, FileOutputStream, OutputStream}
import org.apache.spark.SparkContext
import com.typesafe.config.{Config, ConfigFactory}
import org.deeplearning4j.util.ModelSerializer
import org.apache.hadoop.fs.{FileSystem, Path}

import java.net.URI
import java.nio.file.{Files, Paths}
import org.apache.spark.SparkContext
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer

object FileUtil {

  private val logger = LoggerFactory.getLogger(SparkLLMTraining.getClass)

  def getFileContentAsList(sc: SparkContext, directoryPath: String): util.ArrayList[String] = {
    val linesBuffer = new util.ArrayList[String]()
    val hadoopConf = sc.hadoopConfiguration

    // Check if the path is HDFS or local
    if (directoryPath.startsWith("s3://")) {
      // Handle HDFS case
      val fs = FileSystem.get(new java.net.URI(directoryPath), hadoopConf)
      val path = new Path(directoryPath)
      if (fs.exists(path) && fs.isDirectory(path)) {
        val fileStatuses = fs.listStatus(path)

        fileStatuses.filter(_.isFile).foreach { fileStatus =>
          val filePath = fileStatus.getPath
          // Read each file from HDFS
          Using(Source.fromInputStream(fs.open(filePath))) { source =>
            source.getLines().forEachRemaining(line => linesBuffer.add(line))
          }.recover {
            case e: Exception =>
              logger.error(s"An error occurred while reading the HDFS file ${filePath.getName}: ${e.getMessage}")
          }
        }
      } else {
        logger.error(s"The provided HDFS path is not a directory or does not exist: $directoryPath")
      }
    } else {
      // Handle local file system case
      val dir = new File(directoryPath)
      if (dir.exists && dir.isDirectory) {
        val files = dir.listFiles.filter(_.isFile).toList

        files.foreach { file =>
          Using(Source.fromFile(file)) { source =>
            source.getLines().forEachRemaining(line => linesBuffer.add(line))
          }.recover {
            case e: Exception =>
              logger.error(s"An error occurred while reading the local file ${file.getName}: ${e.getMessage}")
          }
        }
      } else {
        logger.error(s"The provided local path is not a directory: $directoryPath")
      }
    }

    linesBuffer
  }


def loadEmbeddings(sc: SparkContext, directoryPath: String): Map[Int, Array[Double]] = {
    val embeddingMapBuilder = mutable.Map[Int, Array[Double]]()
    val hadoopConf = sc.hadoopConfiguration

    // Check if the path is HDFS or local
    if (directoryPath.startsWith("s3://")) {
      // Handle HDFS case
      val fs = FileSystem.get(new java.net.URI(directoryPath), hadoopConf)
      val path = new Path(directoryPath)
      if (fs.exists(path) && fs.isDirectory(path)) {
        val fileStatuses = fs.listStatus(path)

        fileStatuses.filter(_.isFile).foreach { fileStatus =>
          val filePath = fileStatus.getPath
          // Read each file from HDFS
          Using(Source.fromInputStream(fs.open(filePath))) { source =>
            parseFileContent(source, embeddingMapBuilder)
          }.recover {
            case e: Exception =>
              logger.error(s"An error occurred while reading the HDFS file ${filePath.getName}: ${e.getMessage}")
          }
        }
      } else {
        logger.error(s"The provided HDFS path is not a directory or does not exist: $directoryPath")
      }
    } else {
      // Handle local file system case
      val dir = new File(directoryPath)
      if (dir.exists && dir.isDirectory) {
        val files = dir.listFiles.filter(_.isFile).toList

        files.foreach { file =>
          Using(Source.fromFile(file)) { source =>
            parseFileContent(source, embeddingMapBuilder)
          }.recover {
            case e: Exception =>
              logger.error(s"An error occurred while reading the local file ${file.getName}: ${e.getMessage}")
          }
        }
      } else {
        logger.error(s"The provided local path is not a directory: $directoryPath")
      }
    }

    embeddingMapBuilder.toMap
  }

  private def parseFileContent(source: Source, embeddingMapBuilder: mutable.Map[Int, Array[Double]]): Unit = {
    // Regex to extract the ID and the embedding vector
    val pattern = """word:\s+\S+\s+id:\s+(\d+)\s+freq:\s+\d+\s+\[([^\]]+)\]""".r

    for (line <- source.getLines()) {
      line match {
        case pattern(idString, embeddingString) =>
          val id = idString.toInt
          val embedding = embeddingString.split(",").map(_.trim.toDouble)
          embeddingMapBuilder(id) = embedding
        case _ => // Ignore lines that don't match the expected pattern
      }
    }
  }

  // Load configuration based on path (either local or S3)
  def loadConfig(sc: SparkContext, confPath: String): Config = {
    if (confPath.startsWith("s3://")) {
      // Read from S3 using SparkContext
      val configData = sc.textFile(confPath).collect().mkString("\n")
      ConfigFactory.parseString(configData)
    } else {
      // Read locally using java.io.File
      ConfigFactory.parseFile(new File(confPath))
    }
  }

  // Save model either locally or on S3
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



}
