import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import java.io.File

class Word2VecMRTests extends AnyFlatSpec with Matchers {

  // Test for splitArrayOnToken function
  "splitArrayOnToken" should "correctly split the array on the given token" in {
    val input: Array[Int] = Array(1, 2, 3, 13, 4, 5, 6, 13) // Example input with token 13 as the separator
    val expected: List[Array[Int]] = List(Array(1, 2, 3, 13), Array(4, 5, 6, 13)) // Expected output
    val result = splitArrayOnToken(input, 13)
    result(0) shouldEqual expected(0)
    result(1) shouldEqual expected(1)
  }

  // Test for TokenizedSequenceIterator
  "TokenizedSequenceIterator" should "correctly generate sequences from token arrays" in {
    val tokens = List(Array(1, 2, 3), Array(4, 5, 6)) // Example tokens
    val iterator = new TokenizedSequenceIterator(tokens)

    // Check that sequences are generated correctly
    iterator.hasMoreSequences shouldBe true
    val firstSequence = iterator.nextSequence()
    firstSequence.getElements.size() shouldBe 3

    val secondSequence = iterator.nextSequence()
    secondSequence.getElements.size() shouldBe 3

    iterator.hasMoreSequences shouldBe false
  }

  // The method to compute cosine similarity from CosineSimMR
  def cosineSimilarity(vecA: Array[Double], vecB: Array[Double]): Double =
    val dotProduct = vecA.zip(vecB).map { case (a, b) => a * b }.sum
    val magnitudeA = math.sqrt(vecA.map(a => a * a).sum)
    val magnitudeB = math.sqrt(vecB.map(b => b * b).sum)

    dotProduct / (magnitudeA * magnitudeB)

  // Test for cosineSimilarity function
  "cosineSimilarity" should "correctly calculate the cosine similarity between two vectors" in {
    val vecA = Array(1.0, 2.0, 3.0)
    val vecB = Array(4.0, 5.0, 6.0)
    val similarity = cosineSimilarity(vecA, vecB)

    // Manually calculated cosine similarity for these vectors
    similarity shouldEqual 0.9746318461970762 +- 0.0001
  }

  "Word2VecMR" should "delete the output folder, run, and create _SUCCESS file" in {
    val outputFolder = new File("/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/output")

    // Delete the output folder if it exists
    if (outputFolder.exists()) {
      outputFolder.deleteRecursively()
    }

    // Run the Word2VecMR job
    Word2VecMR.run("/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/input", "/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/output", "/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/application.conf")

    // Check if the _SUCCESS file is created
    val successFile = new File(outputFolder.getAbsolutePath + "/_SUCCESS")
    successFile.exists() shouldEqual true
  }

  "CosineSimMR" should "check for the output folder, delete pair folder, run, and create _SUCCESS file" in {
    val outputFolder = new File("/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/output")
    val pairFolder = new File("/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/pair")

    // Check if the output folder exists, fail if not
    outputFolder.exists() shouldEqual true

    // Delete the pair folder if it exists
    if (pairFolder.exists()) {
      pairFolder.deleteRecursively()
    }

    // Run the CosineSimMR job
    CosineSimMR.run("/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/output", "/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/pair")



    // Check if the _SUCCESS file is created
    val successFile = new File(pairFolder.getAbsolutePath + "/_SUCCESS")
    successFile.exists() shouldEqual true
  }

  // Helper method to recursively delete directories
  implicit class RichFile(file: File) {
    def deleteRecursively(): Unit = {
      if (file.isDirectory) {
        file.listFiles().foreach(_.deleteRecursively())
      }
      file.delete()
    }
  }
}