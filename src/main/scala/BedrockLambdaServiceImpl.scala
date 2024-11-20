import lambda.BedrockLambdaServiceGrpc.BedrockLambdaService
import lambda.{InputMessage, OutputMessage}

import scala.concurrent.Future

class BedrockLambdaServiceImpl extends BedrockLambdaService {
  override def invokeBedrockLambda(request: InputMessage): Future[OutputMessage] = {
    // Process the request and return a response
    val input = request.input
    val response = s"Processed input: $input"
    Future.successful(OutputMessage(response))
  }
}
