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
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.nd4j.linalg.factory.Nd4j

import java.io.IOException
import scala.jdk.CollectionConverters.*
import scala.math.sqrt
import scala.collection.mutable

// Mapper class
class W2VMapper extends Mapper[LongWritable, Text, Text, Text]:

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
class W2VReducer extends Reducer[Text, Text, Text, Text]:

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

//// Cosine Mapper class
//class CosineSimilarityMapper extends Mapper[LongWritable, Text, Text, Text]:
//
//  override def map(key: LongWritable, value: Text, context: Mapper[LongWritable, Text, Text, Text]#Context): Unit =
//    context.write(new Text("cosine"), value)
//
//
//// Cosine Reducer class
//class CosineSimilarityReducer extends Reducer[Text, Text, Text, Text]:
//
//  // Compute cosine similarity between two vectors
//  def cosineSimilarity(vec1: Array[Float], vec2: Array[Float]): Float = 
//    val dotProduct = vec1.zip(vec2).map { case (v1, v2) => v1 * v2 }.sum
//    val norm1 = sqrt(vec1.map(v => v * v).sum).toFloat
//    val norm2 = sqrt(vec2.map(v => v * v).sum).toFloat
//    if (norm1 == 0 || norm2 == 0) 0f else dotProduct / (norm1 * norm2)
//
//  @throws[IOException]
//  @throws[InterruptedException]
//  override def reduce(key: Text, values: java.lang.Iterable[Text], context: Reducer[Text, Text, Text, Text]#Context): Unit = 
//    // Map to store word embeddings (key: word, value: embedding vector)
//    val wordEmbeddings = mutable.Map[String, Array[Float]]()
//
//    // Iterate through all the values (which contain word and embeddings in "word\t[embedding]" format)
//    for (value <- values.asScala) {
//      // Split the value into word and embedding part
//      val tokens = value.toString.split("\t")
//      if (tokens.length == 2) {
//        val word = tokens(0)
//        val embeddingVector: Array[Float] = tokens(1).stripPrefix("[").stripSuffix("]").split(",").map(_.toFloat)
//        wordEmbeddings(word) = embeddingVector
//      }
//    }
//
//    // Debugging: Print keys (words) and their embeddings
//    println(key.toString + " " + wordEmbeddings.keys.toArray.mkString("Array(", ", ", ")"))
//    wordEmbeddings.foreach { case (word, embeddingArray) =>
//      println(word + " " + embeddingArray.mkString("Array(", ", ", ")"))
//    }
//
//    // Compute the top 3 closest words for each word
//    val top3ClosestWords = wordEmbeddings.map { case (word, embedding) =>
//      // Compute cosine similarity to all other words
//      val similarities = wordEmbeddings
//        .view
//        .filterKeys(_ != word) // Exclude the word itself
//        .map { case (otherWord, otherEmbedding) =>
//          val similarity = cosineSimilarity(embedding, otherEmbedding)
//          (otherWord, similarity)
//        }.toMap
//
//      // Sort by similarity and take the top 3 closest words
//      val top3Closest = similarities.toList.sortBy(-_._2).take(3)
//      (word, top3Closest)
//    }
//
//    // Emit the results
//    for ((word, closestWords) <- top3ClosestWords) {
//      val closestWordsStr = closestWords.map { case (closeWord, similarity) => s"$closeWord($similarity)" }.mkString(", ")
//      context.write(new Text(word), new Text(closestWordsStr))
//    }

// Driver code
object Word2VecMR:
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

    job1.setMapperClass(classOf[W2VMapper])
    job1.setReducerClass(classOf[W2VReducer])

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
      System.exit(1)
    } else {
      println("Job failed")
      System.exit(0)
    }

//    val job2 = Job.getInstance(conf, "cosine")
//    job2.setJarByClass(this.getClass)
//    job2.setMapperClass(classOf[CosineSimilarityMapper])
//    job2.setReducerClass(classOf[CosineSimilarityReducer])
//
//    // Input and output key/value types for the second job
//    job2.setOutputKeyClass(classOf[Text])
//    job2.setOutputValueClass(classOf[Text])
//
//    // Use the output of the first job as input for the second job
//    FileInputFormat.addInputPath(job2, new Path(args(1))) // Input from the first job's output
//    FileOutputFormat.setOutputPath(job2, new Path(args(2))) // Final output path
//
//    // Start the job and wait for completion
//    val success2 = job2.submit() // Start job asynchronously
//
//    // Monitor job progress while it runs
//    while (!job2.isComplete) {
//      println(f"Cosine Map progress: ${job2.mapProgress() * 100}%.2f%%")
//      println(f"Cosine Reduce progress: ${job2.reduceProgress() * 100}%.2f%%")
//      Thread.sleep(1000) // Sleep for 5 seconds before checking progress again
//    }
//
//    // Block until the job is done
//    if (job2.waitForCompletion(true)) {
//      println("Cosine Job completed successfully")
//      System.exit(1)
//    } else {
//      println("Job failed")
//      System.exit(0)
//    }