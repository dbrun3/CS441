import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule

import scala.concurrent.{ExecutionContext, Future}
import lambda.{InputMessage, OutputMessage}
import scalaj.http._

import java.util.Base64

case class LambdaInput(input: String)
case class LambdaOutput(output: String)

class BedrockLambdaGrpcClient(url: String)(implicit ec: ExecutionContext) {

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

}
