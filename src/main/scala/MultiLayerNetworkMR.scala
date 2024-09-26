import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingRegistry
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingType
import com.knuddels.jtokkit.api.IntArrayList
import org.apache.hadoop.mapreduce.*
import org.apache.hadoop.fs.Path
import org.apache.hadoop.conf.*
import org.apache.hadoop.io.*
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.conf.Configuration
import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.nd4j.linalg.factory.Nd4j

import java.io.IOException
import scala.jdk.CollectionConverters.*

// Mapper class
class MLNMapper extends Mapper[LongWritable, Text, Text, Text]:

  private val word = new Text()
  private val arr = new Text()

  @throws[IOException]
  @throws[InterruptedException]
  override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, Text]#Context): Unit =
    val line = value.toString
    val lines = line.split("\\n.")
    val javaLines: java.util.List[String] = lines.toList.asJava

    val iter = new CollectionSentenceIterator(javaLines)
    val t = new DefaultTokenizerFactory()
    t.setTokenPreProcessor(new CommonPreprocessor())

    val vec = new Word2Vec.Builder()
      .minWordFrequency(3)
      .layerSize(300)
      .seed(42)
      .windowSize(5)
      .iterate(iter)
      .tokenizerFactory(t)
      .build()

    vec.fit()

    val vocab = vec.getVocab // Get the words in the vocabulary
    val words = vocab.words()

    for (token <- words.asScala) {
      val wordVector = vec.getWordVector(token) // Get the embedding vector for the word
      val tokenId = vocab.wordFor(token).getIndex // Get the token ID (index) of the word

      // println(s"Word: $token, Token ID: $tokenId, Word Vector: ${wordVector.mkString(", ")}")

      word.set(token)
      arr.set(wordVector.mkString(","))
      context.write(word, arr)
    }

// Reducer class
class MLNReducer extends Reducer[Text, Text, Text, Text]:

  @throws[IOException]
  @throws[InterruptedException]
  override def reduce(key: Text, values: java.lang.Iterable[Text], context: Reducer[Text, Text, Text, Text]#Context): Unit =
    var sumVector: Array[Float] = null
    var count = 0

    // Iterate over all values (embedding vectors for the given key)
    for (value <- values.asScala) {
      // Convert the comma-separated string to an array of floats
      val currentVector: Array[Float] = value.toString.stripPrefix("[").stripSuffix("]").split(",").map(_.toFloat)

      // Initialize sumVector if this is the first vector
      if (sumVector == null) {
        sumVector = new Array[Float](currentVector.length)
      }

      // Add the current vector to the sumVector element-wise
      for (i <- currentVector.indices) {
        sumVector(i) += currentVector(i)
      }
      count += 1 // Track the number of vectors
    }

    // Compute the average vector
    if (count > 0) {
      for (i <- sumVector.indices) {
        sumVector(i) /= count
      }
    }

    // Convert the averaged array to a comma-separated string
    val averagedVectorString = sumVector.map(_.toString).mkString(",")

    // Emit the key and the averaged vector
    context.write(key, new Text("[" + averagedVectorString + "]"))

// Driver code
object MLNMR:
  def main(args: Array[String]): Unit =

    try {
      println(Nd4j.getBackend)
      println(Nd4j.create(2, 2)) // Create a simple matrix to ensure backend is working
    } catch {
      case e: Exception => e.printStackTrace()
    }

    if (args.length != 2) {
      println("Usage: Word2VecMR <input path> <output path>")
      System.exit(-1)
    }

    val conf = new Configuration

    val job1 = Job.getInstance(conf, "word2vec")
    job1.setJarByClass(Word2VecMR.getClass)
    job1.setNumReduceTasks(1)  // Adjust the number of reducers

    conf.set("mapreduce.map.log.level", "DEBUG")
    conf.set("mapreduce.reduce.log.level", "DEBUG")

    job1.setMapperClass(classOf[MLNMapper])
    job1.setReducerClass(classOf[MLNReducer])

    job1.setMapOutputKeyClass(classOf[Text])
    job1.setMapOutputValueClass(classOf[Text])

    job1.setOutputKeyClass(classOf[Text])
    job1.setOutputValueClass(classOf[Text])

    FileInputFormat.addInputPath(job1, new Path(args(0)))
    FileOutputFormat.setOutputPath(job1, new Path(args(1)))

    println("Job started...")

    // Start the job and wait for completion
    val success1 = job1.submit() // Start job asynchronously

    // Monitor job progress while it runs
    while (!job1.isComplete) {
      println(f"Word2Vec Map progress: ${job1.mapProgress() * 100}%.2f%%")
      println(f"Word2Vec Reduce progress: ${job1.reduceProgress() * 100}%.2f%%")
      Thread.sleep(1000) // Sleep for 5 seconds before checking progress again
    }

    // Block until the job is done
    if (job1.waitForCompletion(true)) {
      println("Word2Vec Job completed successfully")
      System.exit(0)
    } else {
      println("Job failed")
      System.exit(1)
    }