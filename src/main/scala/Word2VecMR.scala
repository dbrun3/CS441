//import com.knuddels.jtokkit.Encodings
//import com.knuddels.jtokkit.api.EncodingRegistry
//import com.knuddels.jtokkit.api.Encoding
//import com.knuddels.jtokkit.api.EncodingType
//import com.knuddels.jtokkit.api.IntArrayList
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
import org.deeplearning4j.text.sentenceiterator.{CollectionSentenceIterator, LineSentenceIterator}
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory

import java.io.IOException
import scala.jdk.CollectionConverters.*

// Mapper class
class W2VMapper extends Mapper[LongWritable, Text, Text, Text]:

  private val word = new Text()
  private val arr = new Text()

  @throws[IOException]
  @throws[InterruptedException]
  override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, Text]#Context): Unit =
    val line = value.toString
    val lines = line.split("\\r?\\n")
    val javaLines: java.util.List[String] = lines.toList.asJava

    val iter = new CollectionSentenceIterator(javaLines)
    val t = new DefaultTokenizerFactory()
    t.setTokenPreProcessor(new CommonPreprocessor())

    val vec = new Word2Vec.Builder()
      .minWordFrequency(1)
      .layerSize(100)
      .seed(42)
      .windowSize(5)
      .iterate(iter)
      .tokenizerFactory(t)
      .build()

    vec.fit()

    // Emit the word vectors (embeddings) for each word
    val vocab = vec.getVocab.words() // Get the words in the vocabulary
    for (token <- vocab.asScala) {
      val wordVector = vec.getWordVector(token) // Get the embedding vector for the word
      word.set(token)
      arr.set(wordVector.mkString(","))
      context.write(word, arr)
    }

// Reducer class
class W2VReducer extends Reducer[Text, Text, Text, Text]:

  @throws[IOException]
  @throws[InterruptedException]
  protected def reduce(key: Text, values: Text, context: Reducer[Text, Text, Text, Text]#Context): Unit = {
    // Initialize variables for aggregation

    println(s"Reducer invoked for word: ${key.toString}")

    var sumVector: Array[Float] = null
    var count = 0

    for (arrayWritable <- values.toString.split(",")) {
      
      val csvString = arrayWritable
      val currentVector: Array[Float] = csvString.split(",").map(_.toFloat)

      for (i <- 0 until arrayWritable.length) {
        currentVector(i) = arrayWritable(i).asInstanceOf[FloatWritable].get
      }
      // Initialize sumVector on the first iteration
      if (sumVector == null) sumVector = new Array[Float](currentVector.length)
      // Sum up the vectors element-wise
      for (i <- currentVector.indices) {
        sumVector(i) += currentVector(i)
      }
      count += 1 // Track how many vectors we are averaging

    }
    // Compute the average of the embeddings (element-wise division)
    for (i <- sumVector.indices) {
      sumVector(i) /= count
    }

    // Convert the averaged array (sumVector) into a comma-separated string
    val averagedVectorString = sumVector.map(v => v.toString).mkString(",")

    // Create a Text object from the averaged vector string
    val averagedText = new Text("[" + averagedVectorString + "]")

    // Emit the Text representation of the averaged vector
    context.write(key, averagedText)
  }

// Driver code
object Word2VecMR:
  def main(args: Array[String]): Unit =
    if (args.length != 2) {
      println("Usage: Word2VecMR <input path> <output path>")
      System.exit(-1)
    }

    val conf = new Configuration

    val job = Job.getInstance(conf, "word2vec")
    job.setJarByClass(Word2VecMR.getClass)
    job.setNumReduceTasks(1)  // Adjust the number of reducers

    conf.set("mapreduce.map.log.level", "DEBUG")
    conf.set("mapreduce.reduce.log.level", "DEBUG")

    job.setMapperClass(classOf[W2VMapper])
    job.setReducerClass(classOf[W2VReducer])

    job.setMapOutputKeyClass(classOf[Text])
    job.setMapOutputValueClass(classOf[Text])

    job.setOutputKeyClass(classOf[Text])
    job.setOutputValueClass(classOf[Text])

    FileInputFormat.addInputPath(job, new Path(args(0)))
    FileOutputFormat.setOutputPath(job, new Path(args(1)))

    println("Job started...")

    // Start the job and wait for completion
    val success = job.submit() // Start job asynchronously

    // Monitor job progress while it runs
    var jobInProgress = true
    while (!job.isComplete) {
      println(f"Map progress: ${job.mapProgress() * 100}%.2f%%")
      println(f"Reduce progress: ${job.reduceProgress() * 100}%.2f%%")
      Thread.sleep(1000) // Sleep for 5 seconds before checking progress again
    }

    // Block until the job is done
    if (job.waitForCompletion(true)) {
      println("Job completed successfully")

      // Access counters after job completion
      val counters = job.getCounters
      val mapperInputRecords = counters.findCounter("org.apache.hadoop.mapreduce.TaskCounter", "MAP_INPUT_RECORDS").getValue
      val mapperOutputRecords = counters.findCounter("org.apache.hadoop.mapreduce.TaskCounter", "MAP_OUTPUT_RECORDS").getValue
      val reducerInputRecords = counters.findCounter("org.apache.hadoop.mapreduce.TaskCounter", "REDUCE_INPUT_RECORDS").getValue
      val reducerOutputRecords = counters.findCounter("org.apache.hadoop.mapreduce.TaskCounter", "REDUCE_OUTPUT_RECORDS").getValue

      // Print the counters
      println(s"Mapper input records: $mapperInputRecords")
      println(s"Mapper output records: $mapperOutputRecords")
      println(s"Reducer input records: $reducerInputRecords")
      println(s"Reducer output records: $reducerOutputRecords")
    } else {
      println("Job failed")
    }

    System.exit(if (job.isSuccessful) 0 else 1)