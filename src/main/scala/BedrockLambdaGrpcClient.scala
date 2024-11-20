import io.grpc.netty.NettyChannelBuilder

import scala.concurrent.{ExecutionContext, Future}
import lambda.{BedrockLambdaServiceGrpc, InputMessage}

class BedrockLambdaGrpcClient(host: String, port: Int)(implicit ec: ExecutionContext) {
  // Create the gRPC channel
  private val channel = NettyChannelBuilder
    .forAddress(host, port) // Connect to the specified host and port
    .usePlaintext() // Use plaintext (no TLS) for simplicity
    .build()

  // Create the gRPC client stub
  private val stub = BedrockLambdaServiceGrpc.stub(channel)

  // Function to send a message to the gRPC server
  def send(message: String): Future[String] = {
    val request = InputMessage(message) // Build the request
    stub.invokeBedrockLambda(request).map(_.output) // Send the request and map the response
  }

  // Shutdown the client
  def shutdown(): Unit = {
    channel.shutdown()
  }
}
