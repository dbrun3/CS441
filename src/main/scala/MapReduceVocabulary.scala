import org.apache.hadoop.fs.Path
import org.apache.hadoop.conf.*
import org.apache.hadoop.io.*
import org.apache.hadoop.util.*
import org.apache.hadoop.mapred.*

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


object MapReduceVocabulary:
  class Map extends MapReduceBase with Mapper[LongWritable, Text, Text, IntWritable]:

    private var encoding: Encoding = uninitialized

    override def configure(job: JobConf): Unit =
      val registry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()
      encoding = registry.getEncoding(EncodingType.CL100K_BASE) // Using CL100K_BASE for encoding

    @throws[IOException]
    override def map(key: LongWritable, value: Text, output: OutputCollector[Text, IntWritable], reporter: Reporter): Unit =
      val line = value.toString

      val encodedTokens: IntArrayList = encoding.encode(line)
      val tokens: Array[String] = encoding.decode(encodedTokens).split(" ")

      for (i <- 0 until encodedTokens.size()) {
        val tokenId = encodedTokens.get(i) // Token ID
        val token = tokens(i)

        // Emit the token string and its corresponding token ID
        output.collect(new Text(token), new IntWritable(tokenId))
      }



  class Reduce extends MapReduceBase with Reducer[Text, IntWritable, Text, IntWritable]:
    override def reduce(key: Text, values: util.Iterator[IntWritable], output: OutputCollector[Text, IntWritable], reporter: Reporter): Unit =
//      val sum = values.asScala.reduce((valueOne, valueTwo) => new IntWritable(valueOne.get() + valueTwo.get()))
//      output.collect(key,  new IntWritable(sum.get())) TODO: New reducing for tensor
        val out = 0

  @main def runMapReduce(inputPath: String, outputPath: String) =
    val conf: JobConf = new JobConf(this.getClass)
    conf.setJobName("WordCount")
    conf.set("fs.defaultFS", "local")
    conf.set("mapreduce.job.maps", "1")
    conf.set("mapreduce.job.reduces", "1")
    conf.setOutputKeyClass(classOf[Text])
    conf.setOutputValueClass(classOf[IntWritable])
    conf.setMapperClass(classOf[Map])
    conf.setCombinerClass(classOf[Reduce])
    conf.setReducerClass(classOf[Reduce])
    conf.setInputFormat(classOf[TextInputFormat])
    conf.setOutputFormat(classOf[TextOutputFormat[Text, IntWritable]])
    FileInputFormat.setInputPaths(conf, new Path(inputPath))
    FileOutputFormat.setOutputPath(conf, new Path(outputPath))
    JobClient.runJob(conf)