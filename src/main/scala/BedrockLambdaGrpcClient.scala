import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.typesafe.config.ConfigFactory
import io.grpc.netty.NettyChannelBuilder

import scala.concurrent.{ExecutionContext, Future}
import lambda.{BedrockLambdaServiceGrpc, InputMessage, OutputMessage}
import scalaj.http._

import java.util.Base64

case class LambdaInput(input: String)
case class LambdaOutput(output: String)

class BedrockLambdaGrpcClient(url: String)(implicit ec: ExecutionContext) {

  private val config = ConfigFactory.load()
  private val testPort = config.getInt("server.testPort")
  private val testHost = config.getString("server.testServer")

  private val objectMapper = new ObjectMapper()
  objectMapper.registerModule(DefaultScalaModule)

  // Function to send a message to the "gRPC server" aka my lambda
  def send(message: String): Future[String] = {
    Future {
      // Build the InputMessage protobuf object
      val inputMessage = InputMessage(input = message)

      // Serialize the InputMessage to bytes
      val serializedBytes:Array[Byte] = inputMessage.toByteArray
      val base64Payload: String = Base64.getEncoder.encodeToString(serializedBytes)
      val lambdaInput = LambdaInput(input = base64Payload)
      val payload = objectMapper.writeValueAsString(lambdaInput)

      // Make the HTTP POST request
      val response: HttpResponse[String] = Http(url)
        .postData(s"$payload") // Use the Base64 payload as the request body
        .header("Content-Type", "application/json") // Set appropriate headers
        .header("Accept", "application/json")
        .timeout(connTimeoutMs = 5000, readTimeoutMs = 30000) // Adjust as needed
        .asString

      // Check for successful response
      if (response.is2xx) {
        val responseBody = response.body
        val parsedResponse = objectMapper.readValue(responseBody, classOf[LambdaOutput])

        // Decode the response body from Base64
        val decodedBytes = Base64.getDecoder.decode(parsedResponse.output)
        val outputMessage = OutputMessage.parseFrom(decodedBytes)
        outputMessage.output
      } else {
        throw new RuntimeException(s"Failed to send request: ${response.code}, ${response.body}")
      }
    }
  }

  private val channel = NettyChannelBuilder
    .forAddress(testHost, testPort) // Connect to the specified host and port
    .usePlaintext() // Use plaintext (no TLS) for simplicity
    .build()

  // Create the gRPC client stub
  private val stub = BedrockLambdaServiceGrpc.stub(channel)

  // Function to send a message to the gRPC test server
  def sendTestGRPC(message: String): Future[String] = {
    val request = InputMessage(message) // Build the request
    stub.invokeBedrockLambda(request).map(_.output) // Send the request and map the response
  }
  // Shutdown the client
  def shutdown(): Unit = {
    channel.shutdown()
  }

}
