import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, RNNFormat}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

object LLMModel {
  def createModel(embeddingDim: Int, hiddenSize: Int, vocabSize: Int, learnRate: Double): MultiLayerNetwork = {

    // Define the updated model configuration for overfitting
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .weightInit(WeightInit.XAVIER)
      .updater(new Adam(learnRate)) // Use a small learning rate
      .l2(0.0) // Minimize any additional regularization effects
      .list()
      .layer(0, new LSTM.Builder()
        .nIn(embeddingDim)
        .nOut(hiddenSize)
        .activation(Activation.TANH)
        .dropOut(0.1) // Add dropout with a low rate to avoid excessive information loss
        .build())
      .layer(1, new RnnOutputLayer.Builder(LossFunction.MCXENT)
        .activation(Activation.SOFTMAX)
        .nIn(hiddenSize)
        .nOut(vocabSize)
        .build())
      .build()

    // Initialize the model
    val model = new MultiLayerNetwork(conf)
    model.init()

    model  // Return the created model
  }
}
