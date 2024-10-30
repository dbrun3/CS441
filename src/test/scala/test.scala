import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import java.io.File

// /home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/input /home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/model/LLM_Spark_Model.zip /home/dbrun3/Desktop/441/CS441_Fall2024/output/embeddings /home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/application.conf

class HW2Test extends AnyFlatSpec with Matchers {

  // Helper method to delete all files in a directory
  def clearDirectory(directoryPath: String): Unit = {
    val directory = new File(directoryPath)
    if (directory.exists && directory.isDirectory) {
      directory.listFiles().foreach(_.delete())
    }
  }

  "SparkLLMTraining" should "create a model.zip upon success" in {

    // Replace with your paths before testing obv
    val args: Array[String] = Array(
      "/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/input",
      "/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/model/LLM_Spark_Model.zip",
      "/home/dbrun3/Desktop/441/CS441_Fall2024/output/embeddings",
      "/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/test.conf"
    )
    val outputPath = "/home/dbrun3/Desktop/441/CS441_Fall2024/src/main/resources/model/" // Replace with your directory path

    // Step 1: Clear the output directory
    clearDirectory(outputPath)
    val directory = new File(outputPath)

    val filesBeforeRun = directory.listFiles()
    filesBeforeRun shouldBe empty

    // Step 2: Run training
    SparkLLMTraining.main(args)

    // Step 3: Check if any files were created in the directory
    val filesAfterRun = directory.listFiles()

    // Clear directory afterward
    clearDirectory(outputPath)

    // Assert that there is at least one file in the directory
    filesAfterRun should not be empty
  }
}