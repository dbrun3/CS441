import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, RNNFormat}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

object LLMModel {
  def createModel(embeddingDim: Int, hiddenSize: Int): MultiLayerNetwork = {

    // Define the model configuration
    val conf = new NeuralNetConfiguration.Builder()
      .seed(42)  // Random seed for reproducibility
      .list()

      .layer(new LSTM.Builder()
        .nIn(embeddingDim)  // Input size is the embedding dimension (100)
        .nOut(hiddenSize)  // Number of units in the LSTM layer
        .activation(Activation.TANH)
        .build())

      // Output layer for predicting the next word (the output embedding dimension)
      .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)  // MSE because you're predicting embeddings
        .activation(Activation.IDENTITY)  // Linear output for regression
        .nIn(hiddenSize)  // Input to this layer comes from LSTM hidden units
        .nOut(embeddingDim)  // Output is...
        .build())

      .setInputType(InputType.recurrent(embeddingDim))
      .build()

    // Initialize the model
    val model = new MultiLayerNetwork(conf)
    model.init()

    model  // Return the created model
  }
}