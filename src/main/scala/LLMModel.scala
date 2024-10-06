import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

object LLMModel {
  def createModel(embeddingDim: Int, hiddenUnits: Int, numClasses: Int): MultiLayerNetwork = {

    // Define the model configuration
    val conf = new NeuralNetConfiguration.Builder()
      .seed(42)  // Random seed for reproducibility
      .list()

      // wtf

      // The model expects sequential input (recurrent)
      .setInputType(InputType.recurrent(embeddingDim))
      .build()

    // Initialize the model
    val model = new MultiLayerNetwork(conf)
    model.init()

    model  // Return the created model
  }
}