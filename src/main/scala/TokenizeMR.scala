import org.apache.hadoop.fs.Path
import org.apache.hadoop.conf.*
import org.apache.hadoop.io.*
import org.apache.hadoop.mapreduce.*

import java.io.IOException
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{Encoding, EncodingRegistry, EncodingType, IntArrayList}
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat

import scala. jdk. CollectionConverters. IterableHasAsScala
/*
 *   TokenizeMR
 *   Parallelizes BPE tokenization of text corpus
 */

class TokenMap extends Mapper[LongWritable, Text, Text, IntWritable]:

  private final val one = new IntWritable(1)
  private val word = new Text()

  private val registry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()
  private val encoding: Encoding = registry.getEncoding(EncodingType.CL100K_BASE) // Using CL100K_BASE for encoding

  @throws[IOException]
  override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, IntWritable]#Context): Unit =
    val line = value.toString
    val encodedTokens: Array[Int] = encoding.encode(line).toArray

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

object TokenizeMR:
  def run(input: String, output: String): Unit =

    val conf = new Configuration

    val job1 = Job.getInstance(conf, "Tokenize")
    job1.setJarByClass(TokenizeMR.getClass)
    job1.setNumReduceTasks(1) // Adjust the number of reducers

    conf.set("mapreduce.map.log.level", "DEBUG")
    conf.set("mapreduce.reduce.log.level", "DEBUG")


    job1.setMapperClass(classOf[TokenMap])
    job1.setReducerClass(classOf[TokenReduce])

    job1.setMapOutputKeyClass(classOf[Text])
    job1.setMapOutputValueClass(classOf[IntWritable])

    job1.setOutputKeyClass(classOf[Text])
    job1.setOutputValueClass(classOf[Text])

    FileInputFormat.addInputPath(job1, new Path(input))
    FileOutputFormat.setOutputPath(job1, new Path(output))

    println("Job started...")

    // Start the job and wait for completion
    val success1 = job1.submit() // Start job asynchronously

    // Block until the job is done
    if (job1.waitForCompletion(true)) {
      println("Token Job completed successfully")
      System.exit(0)
    } else {
      println("Job failed")
      System.exit(1)
    }