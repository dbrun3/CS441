import JsonFormats.chatInputFormat
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import spray.json._
import com.typesafe.config.ConfigFactory
import org.scalatest.concurrent.ScalaFutures


class HW3Test extends AnyFlatSpec with Matchers with ScalaFutures {

  "ChatInput JSON serialization and deserialization" should "work correctly" in {
    val chatInput = ChatInput("Hello")
    val json = chatInput.toJson.compactPrint
    val expectedJson = """{"input":"Hello"}"""
    json shouldEqual expectedJson

    val deserializedChatInput = json.parseJson.convertTo[ChatInput]
    deserializedChatInput shouldEqual chatInput
  }

  "Server configuration" should "load port from application.conf" in {
    val config = ConfigFactory.load()
    val port = config.getInt("server.port")
    port should be > 0
  }
}