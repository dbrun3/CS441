import JsonFormats.chatInputFormat
import akka.actor.typed.ActorSystem
import akka.actor.typed.scaladsl.Behaviors
import akka.http.scaladsl.Http
import akka.http.scaladsl.model._
import akka.http.scaladsl.server.Directives._
import akka.http.scaladsl.marshallers.sprayjson.SprayJsonSupport._
import akka.stream.SystemMaterializer
import com.typesafe.config.ConfigFactory
import spray.json._

import scala.concurrent.ExecutionContext
import scala.io.StdIn
import scala.util.{Failure, Success}

// JSON model and format
case class ChatInput(input: String)

object JsonFormats extends DefaultJsonProtocol {
  implicit val chatInputFormat: RootJsonFormat[ChatInput] = jsonFormat1(ChatInput)
}

object AkkaServer {

  private val config = ConfigFactory.load()
  private val port = config.getInt("server.port")

  def run(args: Array[String]): Unit = {

    val testingContext = args(0) == "test"

    implicit val system: ActorSystem[Nothing] = ActorSystem(Behaviors.empty, "AkkaHttpServer")
    implicit val materializer = SystemMaterializer(system).materializer
    implicit val executionContext: ExecutionContext = system.executionContext

    // Create the gRPC client
    val grpcClient = new BedrockLambdaGrpcClient(args(0))

    // Define the route
    val route =
      path("chat") {
        post {
          entity(as[ChatInput]) { chatInput =>
            // Call the gRPC client's `send` method
            onComplete(if(!testingContext) grpcClient.send(chatInput.input) else grpcClient.sendTestGRPC(chatInput.input)) {
              case Success(response) =>
                complete(HttpEntity(ContentTypes.`application/json`, s"""{"response": "$response"}"""))
              case Failure(exception) =>
                complete(HttpEntity(ContentTypes.`application/json`, s"""{"error": "${exception.getMessage}"}"""))
            }
          }
        }
      }

    // Start the server
    val bindingFuture = Http().newServerAt("0.0.0.0", port).bind(route)

    println(s"Server online at http://0.0.0.0:$port/\nPress RETURN to stop...")
    StdIn.readLine() // Let it run until user presses return

    bindingFuture
      .flatMap(_.unbind())
      .onComplete(_ => {
        system.terminate()
      })
  }
}
