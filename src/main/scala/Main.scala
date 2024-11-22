object Main {
  def main(args: Array[String]): Unit = {
    if (args.isEmpty) {
      println("Please specify which server to run: main or test")
      sys.exit(1)
    }

    args(0).toLowerCase match {
      case "main" =>
        println("Starting Main Akka Server...")

        if(args.tail.length != 1) {
          println("Main server args <url>")
        } else {
          AkkaServer.run(args.tail)
        }
      case "test" =>
        println("Starting gRPC Test Server...")
        BedrockLambdaTestServer.run(args.tail)
      case _ =>
        println(s"Unknown server: ${args(0)}")
        sys.exit(1)
    }
  }
}
