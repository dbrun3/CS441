import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{Encoding, EncodingRegistry, EncodingType, IntArrayList}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{LongWritable, Text, Writable}
import org.apache.hadoop.mapreduce.lib.input.{FileInputFormat, TextInputFormat}
import org.apache.hadoop.mapreduce.lib.output.{FileOutputFormat, TextOutputFormat}
import org.apache.hadoop.mapreduce.{Job, Mapper, Reducer}

import java.io.{File, IOException}
import scala.compiletime.uninitialized
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.DataSet
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.nd4j.linalg.activations.Activation
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.learning.config.Adam

object DL4JModel:
  def createModel(vocabSize: Int, embeddingSize: Int, numClasses: Int): MultiLayerNetwork =

    // Define the network configuration
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .weightInit(WeightInit.XAVIER)            // Xavier weight initialization
      .updater(new Adam(0.001))                 // Adam optimizer with a learning rate of 0.001
      .list()
      .layer(0, new EmbeddingLayer.Builder()
        .nIn(vocabSize)                        // Input is the size of the vocabulary (token indices)
        .nOut(embeddingSize)                   // Output is the embedding size (dimension of each token embedding)
        .build()
      )
      .layer(1, new DenseLayer.Builder()
        .nIn(embeddingSize)                    // Input to this layer is the embedding size
        .nOut(128)                             // Number of hidden units in this dense layer
        .activation(Activation.RELU)           // Use ReLU activation function
        .build()
      )
      .layer(2, new DenseLayer.Builder()
        .nIn(128)                              // Input is the previous layer size
        .nOut(64)                              // Number of hidden units in this dense layer
        .activation(Activation.RELU)           // Another ReLU activation function
        .build()
      )
      .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)  // Multi-class Cross Entropy Loss
        .nIn(64)                              // Input is from the previous dense layer (64 units)
        .nOut(numClasses)                     // Output is the number of classes (for classification)
        .activation(Activation.SOFTMAX)       // Softmax activation for multi-class classification
        .build()
      )
      .build()

    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
    model.init()
    model

object MapReduceTensor:
  class EmbeddingMapper extends Mapper[LongWritable, Text, Text, Text]:
    var model: MultiLayerNetwork = uninitialized
    private var encoding: Encoding = uninitialized
    private val registry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()
    encoding = registry.getEncoding(EncodingType.CL100K_BASE) // Using CL100K_BASE for encoding

    @throws[IOException]
    @throws[InterruptedException]
    override def setup(context: Mapper[LongWritable, Text, Text, Text]#Context): Unit =
      val vocabSize = context.getConfiguration.get("vocabSize").toInt
      val embeddingSize = context.getConfiguration.get("embeddingSize").toInt
      val numClasses = context.getConfiguration.get("numClasses").toInt
      // Initialize the DL4J model
      model = DL4JModel.createModel(vocabSize, embeddingSize, numClasses)

    @throws[IOException]
    @throws[InterruptedException]
    protected override def map(key: LongWritable, value: Text, context: Context): Unit =

      // Retrieve the line from input
      val line = value.toString
      val encodedTokens: IntArrayList = encoding.encode(line)
      val tokenIndices: Array[Int] = encodedTokens.toArray

      // Create an INDArray for the input features
      val features: INDArray = Nd4j.create(tokenIndices, Array(tokenIndices.length))
      val numClasses = context.getConfiguration.get("numClasses").toInt
      val label = tokenIndices.last // assuming label is embedded as the last token
      val labels: INDArray = Nd4j.create(numClasses) // create with the number of classes
      labels.putScalar(label, 1.0) // one-hot encode the label at the given index

      // Create DataSet and train the model
      val dataSet: DataSet = new DataSet(features, labels)
      model.fit(dataSet)
      // Emit a status update or other information if needed
      context.write(new Text("Trained on record: " + key), new Text("Model updated"))

    @throws[IOException]
    @throws[InterruptedException]
    protected override def cleanup(context: Mapper[LongWritable, Text, Text, Text]#Context): Unit =
      // Save the trained model at the end of the Map phase
      val modelPath = context.getConfiguration.get("modelPath")
      model.save(new File(modelPath))


  class EmbeddingReducer extends Reducer[LongWritable, Text, Text, Text]:
    val idk = 0


  @main def runMapReduceTensor(inputPath: String, outputPath: String, modelPath: String) = {
    val conf = new Configuration()
    conf.set("vocabSize", "10000")
    conf.set("embeddingSize", "128")
    conf.set("numClasses", "10")
    conf.set("modelPath", modelPath)

    val job = Job.getInstance(conf, "Token Embedding Training")
    job.setJarByClass(classOf[EmbeddingMapper]) // Assuming EmbeddingMapper is your custom mapper

    // Set the Mapper and Reducer classes
    job.setMapperClass(classOf[EmbeddingMapper]) // Your custom mapper class
    job.setReducerClass(classOf[EmbeddingReducer]) // Your custom reducer class

    // Set output key/value classes
    job.setOutputKeyClass(classOf[Text]) // Adjust types as needed
    job.setOutputValueClass(classOf[Text]) // Adjust types as needed

    // Set input/output format classes (e.g., TextInputFormat)
    job.setInputFormatClass(classOf[TextInputFormat]) // Adjust to your format
    job.setOutputFormatClass(classOf[TextOutputFormat[Text, Text]]) // Adjust to your format

    // Set input and output paths
    FileInputFormat.addInputPath(job, new Path(inputPath))
    FileOutputFormat.setOutputPath(job, new Path(outputPath))

    // Wait for job completion
    if (job.waitForCompletion(true)) {
      println("MapReduce job completed successfully.")
    } else {
      println("MapReduce job failed.")
    }
  }
