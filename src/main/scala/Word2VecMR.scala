import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingRegistry
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingType
import com.knuddels.jtokkit.api.IntArrayList
import com.typesafe.config.{Config, ConfigFactory}
import org.apache.hadoop.mapreduce.*
import org.apache.hadoop.fs.Path
import org.apache.hadoop.conf.*
import org.apache.hadoop.io.*
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.conf.Configuration
import org.deeplearning4j.models.sequencevectors.interfaces.SequenceIterator
import org.deeplearning4j.models.word2vec.{VocabWord, Word2Vec}
import org.deeplearning4j.models.sequencevectors.sequence
import org.deeplearning4j.models.sequencevectors.sequence.Sequence
import org.nd4j.linalg.factory.Nd4j

import java.io.IOException
import scala.jdk.CollectionConverters.*

import org.slf4j.LoggerFactory

// Function for splitting encoded arrays by periods
def splitArrayOnToken(arr: Array[Int], separator: Int): List[Array[Int]] =
  val result = scala.collection.mutable.ListBuffer[Array[Int]]()
  val tempBuffer = scala.collection.mutable.ArrayBuffer[Int]()
  for (token <- arr) {
    if (token == separator) {
      tempBuffer += token // Include the separator at the end of the sub-array
      result += tempBuffer.toArray // Add the current sub-array to the result list
      tempBuffer.clear()           // Clear the buffer for the next sub-array
    } else {
      tempBuffer += token // Add the current token to the temporary buffer
    }
  }
  // Add any remaining tokens in the buffer as the last sub-array
  if (tempBuffer.nonEmpty) {
    result += tempBuffer.toArray
  }
  result.toList

// Custom class to handle the sequence of tokens
class TokenizedSequenceIterator(tokensList: List[Array[Int]]) extends SequenceIterator[VocabWord]:

  private var currentIndex: Int = 0

  override def hasMoreSequences: Boolean = currentIndex < tokensList.size

  override def nextSequence(): Sequence[VocabWord] =
    // Create a sequence from the current token array
    val currentTokens = tokensList(currentIndex)
    val sequence = new Sequence[VocabWord]()

    // Convert each token into a VocabWord and add it to the sequence
    currentTokens.foreach { token =>
      val word = new VocabWord(1.0, token.toString) // Default frequency 1.0
      sequence.addElement(word)
    }
    currentIndex += 1 // Move to the next sequence
    sequence

  override def reset(): Unit =
    currentIndex = 0

// Mapper class
// The Mapper class reads input text line by line, encodes the text using BPE (byte pair encoding),
// and splits it into sentences based on periods. It then processes each sentence through a Word2Vec model
// to generate word embeddings. The embeddings and associated tokens are emitted as key-value pairs,
// where the key is the decoded word and the value is the corresponding word vector.
class W2VMapper extends Mapper[LongWritable, Text, Text, Text]:

  private val wordKey = new Text()
  private val arr = new Text()

  private val tokens = new Text()
  private val registry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()
  private val encoding: Encoding = registry.getEncoding(EncodingType.CL100K_BASE) // Using CL100K_BASE for encoding

  private val config: Config = ConfigFactory.load()
  private val minWordFrequency: Int = config.getInt("word2vec.minWordFrequency")
  private val layerSize: Int = config.getInt("word2vec.layerSize")
  private val seed: Long = config.getLong("word2vec.seed")
  private val windowSize: Int = config.getInt("word2vec.windowSize")
  private val epochs: Int = config.getInt("word2vec.epochs")

  @throws[IOException]
  @throws[InterruptedException]
  override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, Text]#Context): Unit =
    val line = value.toString
    val encodedTokens: Array[Int] = encoding.encode(line).toArray

    val encodedSentences: List[Array[Int]] = splitArrayOnToken(encodedTokens, 13) //BPE encoding for period

    val sequenceIterator = new TokenizedSequenceIterator(encodedSentences)

    val vec = new Word2Vec.Builder()
      .minWordFrequency(minWordFrequency)
      .layerSize(layerSize)
      .seed(seed)
      .windowSize(windowSize)
      .epochs(epochs)
      .iterate(sequenceIterator)
      .build()

    vec.fit()

    val vocab = vec.getVocab
    val words = vocab.words().asScala

    words.foreach { word =>
      val tokenId = word.toInt // Convert string back to integer token ID
      val decodedValue: IntArrayList = new IntArrayList()
      decodedValue.add(tokenId)
      val decodedWord = encoding.decode(decodedValue) // Decode to original subword

      val wordVector = vec.getWordVector(word) // Get the embedding vector for the word
      val freq = vec.getVocab.wordFrequency(word)

      wordKey.set("word: " + decodedWord + "\tid: " + tokenId)
      arr.set(freq + "\t" + wordVector.mkString(","))
      context.write(wordKey, arr)
    }

// Reducer class
// The Reducer class collects the word embeddings emitted by the Mapper, aggregates them by summing
// the vectors of the same token, and computes the average vector for each token. The final output
// is the token with its average embedding vector, which represents the word's meaning based on its
// usage in the text.
class W2VReducer extends Reducer[Text, Text, Text, Text]:

  private val out = Text()

  @throws[IOException]
  @throws[InterruptedException]
  override def reduce(key: Text, values: java.lang.Iterable[Text], context: Reducer[Text, Text, Text, Text]#Context): Unit =
    var sumVector: Array[Float] = null
    var count = 0

    // Iterate over all values (embedding vectors for the given key)
    for (value <- values.asScala) {
      // Convert the comma-separated string to an array of floats
      val parts: Array[String] = value.toString.split("\t")
      val freq: Int = parts(0).toInt
      val currentVector: Array[Float] = parts(1).stripPrefix("[").stripSuffix("]").split(",").map(_.toFloat)

      // Initialize sumVector if this is the first vector
      if (sumVector == null) {
        sumVector = new Array[Float](currentVector.length)
      }

      // Add the current vector to the sumVector element-wise
      for (i <- currentVector.indices) {
        sumVector(i) += currentVector(i) * freq
      }
      count += freq // Track the number of vectors
    }

    // Compute the average vector
    if (count > 0) {
      for (i <- sumVector.indices) {
        sumVector(i) /= count
      }
    }

    // Convert the averaged array to a comma-separated string
    val averagedVectorString = sumVector.map(_.toString).mkString(",")

    out.set(key.toString + "\tfreq: " + count)

    // Emit the key and the averaged vector
    context.write(out, new Text("[" + averagedVectorString + "]"))

// Driver code
object Word2VecMR:

  private val logger = LoggerFactory.getLogger(this.getClass)
  
  def run(input: String, output: String): Unit =

    try {
      logger.info("Creating matrix to test backend working")
      logger.info(Nd4j.getBackend.toString)
      logger.info(Nd4j.create(2, 2).toString()) // Create a simple matrix to ensure backend is working
    } catch {
      case e: Exception => logger.error(e.printStackTrace().toString)
    }

    val conf = new Configuration

    val job1 = Job.getInstance(conf, "word2vec")
    job1.setJarByClass(Word2VecMR.getClass)

    conf.set("mapreduce.map.log.level", "DEBUG")
    conf.set("mapreduce.reduce.log.level", "DEBUG")

    job1.setMapperClass(classOf[W2VMapper])
    job1.setReducerClass(classOf[W2VReducer])

    job1.setMapOutputKeyClass(classOf[Text])
    job1.setMapOutputValueClass(classOf[Text])

    job1.setOutputKeyClass(classOf[Text])
    job1.setOutputValueClass(classOf[Text])

    FileInputFormat.addInputPath(job1, new Path(input))
    FileOutputFormat.setOutputPath(job1, new Path(output))

    logger.info("Word2Vec Job started...")

    // Start the job and wait for completion
    val success1 = job1.submit() // Start job asynchronously

    // Block until the job is done
    if (job1.waitForCompletion(true)) {
      logger.info("Word2Vec Job completed successfully")
      // System.exit(0)
    } else {
      logger.info("Job failed")
      System.exit(1)
    }