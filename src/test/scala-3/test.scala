import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers


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

  // Helper method to compute cosine similarity
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
}