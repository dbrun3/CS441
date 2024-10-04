import org.apache.spark.api.java.JavaRDD
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.SparkConf
import java.util


object SlidingWindowSpark {
  def main(args: Array[String]): Unit = {
    // Set up Spark configuration and context
    val conf = new SparkConf().setAppName("Sliding Window Dataset").setMaster("local[*]")
    val sc = new JavaSparkContext(conf)
    // Example input data (could be sentences, tokens, etc.)
    val sentences = Array("The quick brown fox jumps over the lazy dog", "This is another sentence for testing sliding windows")
    // Parallelize the input data (convert array to an RDD)
    val sentenceRDD = sc.parallelize(Arrays.asList(sentences))
    // Apply the sliding window logic to create the dataset
    val slidingWindowDataset = sentenceRDD.flatMap((sentence: String) => createSlidingWindows(sentence, 4).iterator)
    // Collect and print the results (for demonstration)
    slidingWindowDataset.collect.forEach((window: Nothing) => {
      System.out.println("Input: " + Arrays.toString(window.getInput))
      System.out.println("Target: " + window.getTarget)

    })
    // Stop the Spark context
    sc.stop()
  }
}

class SlidingWindowUtils {

  abstract class WindowedData {
    private var input: Array[String]
    private var target: String

    val WindowedData = (i : Array[String], t: String) => {
      input = i;
      target = t;
    }

    def getInput(): Array[String] = {
      input
    }

    def getTarget(): String = {
      target
    }
  }

  // Create sliding windows for a given sentence
  public static List<WindowedData> createSlidingWindows(String sentence, int windowSize) {
    String[] tokens = sentence.split(" ");
    List<WindowedData> windowedDataList = new ArrayList<>();

    // Create sliding windows
    for (int i = 0; i <= tokens.length - windowSize; i++) {
      String[] inputWindow = new String[windowSize];
      System.arraycopy(tokens, i, inputWindow, 0, windowSize);

      String target = tokens[i + windowSize];
      windowedDataList.add(new WindowedData(inputWindow, target));
    }

    return windowedDataList;
  }
}