import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{DenseLayer, EmbeddingLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

object LLMModel {
  def createModel(vocabSize: Int, embeddingDim: Int, numClasses: Int): MultiLayerNetwork = {
    val config: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .list()
      .layer(new EmbeddingLayer.Builder()
        .nIn(vocabSize)         // Input size (vocabulary size)
        .nOut(embeddingDim)      // Output size (embedding dimensions)
        .activation(Activation.IDENTITY)  // No activation function (linear)
        .build())
      .layer(new DenseLayer.Builder()
        .nIn(embeddingDim)
        .nOut(128)
        .activation(Activation.RELU)
        .build())
      .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .activation(Activation.SOFTMAX)
        .nOut(numClasses)
        .build())
      .setInputType(InputType.feedForward(embeddingDim))  // Set input type for embeddings
      .build()

    // Create your LLM model using DL4J
    val model: MultiLayerNetwork = new MultiLayerNetwork(config)
    model.init()
    model
  }
}