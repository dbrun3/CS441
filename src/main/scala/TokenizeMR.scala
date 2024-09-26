import org.apache.hadoop.fs.Path
import org.apache.hadoop.conf.*
import org.apache.hadoop.io.*
import org.apache.hadoop.util.*
import org.apache.hadoop.mapreduce.*

import java.io.IOException
import java.util
import java.util.StringTokenizer
import scala.collection.mutable.HashMap
import scala.jdk.CollectionConverters.*
import scala.compiletime.uninitialized
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingRegistry
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingType
import com.knuddels.jtokkit.api.IntArrayList
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat

object TokenizeMR:
  class TokenMap extends Mapper[LongWritable, Text, Text, IntWritable]:

    private final val one = new IntWritable(1)
    private val word = new Text()

    private val registry = Encodings.newDefaultEncodingRegistry()
    private val encoding = registry.getEncoding(EncodingType.CL100K_BASE) // Using CL100K_BASE for encoding

    @throws[IOException]
    override def map(key: LongWritable, value: Text, context: Mapper[Text, IntWritable, Text, IntWritable]#Context): Unit =
      val line = value.toString

      val encodedTokens = encoding.encode(line).toArray

      encodedTokens.foreach(tokenId => {
        val tokenIdList = new IntArrayList()
        tokenIdList.add(tokenId)
        val token = encoding.decode(tokenIdList)
        word.set(token + "\t" + tokenId)
        context.write(word, one)
      })



  class TokenReduce extends Reducer[Text, IntWritable, Text, IntWritable]:
    override def reduce(key: Text, values: java.lang.Iterable[IntWritable], context: Reducer[Text, IntWritable, Text, IntWritable]#Context): Unit =
      val sum = values.asScala.reduce((valueOne, valueTwo) => new IntWritable(valueOne.get() + valueTwo.get()))
      context.write(key,  new IntWritable(sum.get()))
      val out = 0

object runTMR:
    
  def main(args: Array[String]): Unit =
    
    if (args.length != 2) {
      println("Usage: Word2VecMR <input path> <output path>")
      System.exit(-1)
    }

    val conf = new Configuration

    val job1 = Job.getInstance(conf, "word2vec")
    job1.setJarByClass(Word2VecMR.getClass)
    job1.setNumReduceTasks(1) // Adjust the number of reducers

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
      println(f"Token Map progress: ${job1.mapProgress() * 100}%.2f%%")
      println(f"Token Reduce progress: ${job1.reduceProgress() * 100}%.2f%%")
      Thread.sleep(1000) // Sleep for 5 seconds before checking progress again
    }

    // Block until the job is done
    if (job1.waitForCompletion(true)) {
      println("Token Job completed successfully")
      System.exit(0)
    } else {
      println("Job failed")
      System.exit(1)
    }