import org.apache.hadoop.mapreduce.*
import org.apache.hadoop.fs.Path
import org.apache.hadoop.conf.*
import org.apache.hadoop.io.*
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}
import org.apache.hadoop.conf.Configuration

import org.nd4j.linalg.factory.Nd4j

import java.io.IOException
import scala.jdk.CollectionConverters.*

//Mapper
// The Mapper class processes input text, where each line contains a word and its associated embedding vector.
// It extracts the word and its embedding, and emits them as key-value pairs, where the key is a constant (e.g., "ALL")
// and the value is the word along with its embedding in a tab-separated format.
class WordEmbeddingMapper extends Mapper[LongWritable, Text, Text, Text]:

  @throws[IOException]
  @throws[InterruptedException]
  override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, Text]#Context): Unit =
    val line = value.toString.split("\t")

    val wordWithPrefix = line(0) // "word: This"
    val word = wordWithPrefix.split(":")(1).trim // Extract the word "This"

    // Extract the embedding values from line(3)
    val embeddingString = line(3).replace("[", "").replace("]", "").trim // "-0.1, 0.2, ..."

    // Emit word as key and its embedding as value
    context.write(new Text("ALL"), new Text(word + "\t" + embeddingString))


//Reducer
// The Reducer class gathers all words and their embeddings from the Mapper output. It calculates the cosine similarity
// between each word's embedding and every other word's embedding to find the closest neighboring word based on similarity.
// The final output is the word along with its closest neighbor and the similarity score between their embeddings.
class CosineSimilarityReducer extends Reducer[Text, Text, Text, Text]:

  @throws[IOException]
  @throws[InterruptedException]
  override def reduce(key: Text, values: java.lang.Iterable[Text], context: Reducer[Text, Text, Text, Text]#Context): Unit =
    // Collect all words and embeddings
    val wordEmbeddings = values.asScala.map { value =>
      val parts = value.toString.split("\t")
      val word = parts(0).trim
      val embedding = parts(1).split(",").map(_.trim.toDouble) // Convert embedding to Array[Double]
      (word, embedding)
    }.toList

    // Loop through each word embedding to calculate its closest neighbor
    for (i <- wordEmbeddings.indices) {
      val word1 = wordEmbeddings(i)._1
      val embedding1 = wordEmbeddings(i)._2

      var closestWord: String = ""
      var closestSimilarity: Double = Double.MinValue

      // Compare word1's embedding with every other word's embedding
      for (j <- wordEmbeddings.indices) {
        if (i != j) { // Avoid comparing the word with itself
          val word2 = wordEmbeddings(j)._1
          val embedding2 = wordEmbeddings(j)._2

          val similarity = cosineSimilarity(embedding1, embedding2)

          // Check if this word is the closest neighbor so far
          if (similarity > closestSimilarity) {
            closestSimilarity = similarity
            closestWord = word2
          }
        }
      }

      // Write the closest word for the current word1
      context.write(new Text(s"Closest to $word1: $closestWord"), new Text(s"Similarity: $closestSimilarity"))
    }

  // Helper method to compute cosine similarity
  def cosineSimilarity(vecA: Array[Double], vecB: Array[Double]): Double =
    val dotProduct = vecA.zip(vecB).map { case (a, b) => a * b }.sum
    val magnitudeA = math.sqrt(vecA.map(a => a * a).sum)
    val magnitudeB = math.sqrt(vecB.map(b => b * b).sum)

    dotProduct / (magnitudeA * magnitudeB)

object CosineSimMR:
  def run(input: String, output: String): Unit =

    try {
      println(Nd4j.getBackend)
      println(Nd4j.create(2, 2)) // Create a simple matrix to ensure backend is working
    } catch {
      case e: Exception => e.printStackTrace()
    }

    val conf = new Configuration

    val job1 = Job.getInstance(conf, "cosinesim")
    job1.setJarByClass(CosineSimMR.getClass)
    job1.setNumReduceTasks(1)  // Adjust the number of reducers

    conf.set("mapreduce.map.log.level", "DEBUG")
    conf.set("mapreduce.reduce.log.level", "DEBUG")

    job1.setMapperClass(classOf[WordEmbeddingMapper])
    job1.setReducerClass(classOf[CosineSimilarityReducer])

    job1.setMapOutputKeyClass(classOf[Text])
    job1.setMapOutputValueClass(classOf[Text])

    job1.setOutputKeyClass(classOf[Text])
    job1.setOutputValueClass(classOf[Text])

    FileInputFormat.addInputPath(job1, new Path(input))
    FileOutputFormat.setOutputPath(job1, new Path(output))

    println("Job started...")

    // Start the job and wait for completion
    val success1 = job1.submit() // Start job asynchronously

    // Block until the job is done
    if (job1.waitForCompletion(true)) {
      println("Word2Vec Job completed successfully")
      // System.exit(0)
    } else {
      println("Job failed")
      System.exit(1)
    }