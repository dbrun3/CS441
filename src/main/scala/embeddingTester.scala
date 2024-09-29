//import scala.io.Source
//import scala.math._
//import scala.collection.mutable.Map
//
//object WordEmbeddingSearch {
//  def main(args: Array[String]): Unit = {
//    if (args.length < 1) {
//      println("Usage: WordEmbeddingSearch <file-path>")
//      sys.exit(1)
//    }
//
//    val filePath = args(0)
//    val wordEmbeddings = loadWordEmbeddings(filePath)
//    var input = ""
//    while (input != "quit") {
//      println("Enter a sequence of words with operations (+, -, /) to compute embeddings or similarity (e.g., king - man + woman / queen):")
//      input = scala.io.StdIn.readLine().trim
//
//      if (input.endsWith("!")) {
//        val word = input.stripSuffix("!")
//        if (wordEmbeddings.contains(word)) {
//          val closestWord = findClosestWord(word, wordEmbeddings)
//          println(s"The word closest to '$word' is '$closestWord'")
//        } else {
//          println(s"Word '$word' not found in the embeddings.")
//        }
//      } else {
//        val result = processInput(input, wordEmbeddings)
//
//        result match {
//          case Some((resultEmbedding, Some(similarity))) =>
//            println(s"Cosine similarity: $similarity")
//          case Some((resultEmbedding, None)) =>
//            println(s"Resulting embedding: ${resultEmbedding.mkString("[", ", ", "]")}")
//          case None =>
//            println("Invalid input or word not found in embeddings.")
//        }
//      }
//    }
//  }
//
//  // Function to load word embeddings from the file
//  def loadWordEmbeddings(filePath: String): Map[String, Array[Float]] = {
//    val wordEmbeddings = Map[String, Array[Float]]()
//    for (line <- Source.fromFile(filePath).getLines()) {
//      val parts = line.split("\\s+")
//      if (parts.length > 1) {
//        val word = parts(0)
//        val embedding = parts(1).stripPrefix("[").stripSuffix("]").split(",").map(_.toFloat)
//        wordEmbeddings(word) = embedding
//      }
//    }
//    wordEmbeddings
//  }
//
//  // Function to process the user input and perform vector operations
//  def processInput(input: String, wordEmbeddings: Map[String, Array[Float]]): Option[(Array[Float], Option[Float])] = {
//    // Split input by space and filter out empty tokens
//    val tokens = input.split("\\s+").filter(_.nonEmpty)
//
//    var resultEmbedding: Option[Array[Float]] = None
//    var lastOperation: Option[String] = None
//    var cosineTargetEmbedding: Option[Array[Float]] = None
//
//    tokens.foreach {
//      case token if wordEmbeddings.contains(token) =>
//        // If it's a word, apply the last operation to update the result embedding
//        val currentEmbedding = wordEmbeddings(token)
//        resultEmbedding = resultEmbedding match {
//          case Some(embedding) =>
//            lastOperation match {
//              case Some("+") => Some(addEmbeddings(embedding, currentEmbedding))
//              case Some("-") => Some(subtractEmbeddings(embedding, currentEmbedding))
//              case None      => Some(currentEmbedding) // For the first word
//              case _         => resultEmbedding // If no valid operation
//            }
//          case None => Some(currentEmbedding)
//        }
//      case "+" => lastOperation = Some("+")
//      case "-" => lastOperation = Some("-")
//      case "/" =>
//        // If "/" is encountered, the next word will be the one to compute cosine similarity against
//        lastOperation = None
//        cosineTargetEmbedding = resultEmbedding
//      case other => return None // Invalid token or word not found
//    }
//
//    // If there was a "/" in the input, compute cosine similarity
//    cosineTargetEmbedding match {
//      case Some(targetEmbedding) =>
//        resultEmbedding match {
//          case Some(embedding) =>
//            val similarity = cosineSimilarity(embedding, targetEmbedding)
//            Some((embedding, Some(similarity)))
//          case None => None
//        }
//      case None => resultEmbedding.map(embedding => (embedding, None)) // Just return the embedding if no "/"
//    }
//  }
//
//  // Function to add two embeddings
//  def addEmbeddings(vec1: Array[Float], vec2: Array[Float]): Array[Float] = {
//    vec1.zip(vec2).map { case (v1, v2) => v1 + v2 }
//  }
//
//  // Function to subtract two embeddings
//  def subtractEmbeddings(vec1: Array[Float], vec2: Array[Float]): Array[Float] = {
//    vec1.zip(vec2).map { case (v1, v2) => v1 - v2 }
//  }
//
//  // Function to compute cosine similarity between two vectors
//  def cosineSimilarity(vec1: Array[Float], vec2: Array[Float]): Float = {
//    val dotProduct = vec1.zip(vec2).map { case (v1, v2) => v1 * v2 }.sum
//    val norm1 = sqrt(vec1.map(v => v * v).sum).toFloat
//    val norm2 = sqrt(vec2.map(v => v * v).sum).toFloat
//    dotProduct / (norm1 * norm2)
//  }
//
//  // Function to find the closest word based on cosine similarity
//  def findClosestWord(word: String, wordEmbeddings: Map[String, Array[Float]]): String = {
//    val embedding = wordEmbeddings(word)
//    wordEmbeddings
//      .filterKeys(_ != word) // Exclude the word itself
//      .map { case (otherWord, otherEmbedding) =>
//        (otherWord, cosineSimilarity(embedding, otherEmbedding))
//      }
//      .maxBy(_._2)._1 // Return the word with the highest similarity
//  }
//}
