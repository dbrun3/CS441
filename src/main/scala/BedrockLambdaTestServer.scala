import com.typesafe.config.ConfigFactory
import io.grpc.NameResolverRegistry
import io.grpc.internal.DnsNameResolverProvider
import lambda.BedrockLambdaServiceGrpc.BedrockLambdaService
import io.grpc.netty.NettyServerBuilder

import scala.concurrent.ExecutionContext

object BedrockLambdaTestServer {

  private val config = ConfigFactory.load()
  private val testPort = config.getInt("server.testPort")

  def run(args: Array[String]): Unit = {
    implicit val ec: ExecutionContext = ExecutionContext.global

    NameResolverRegistry.getDefaultRegistry.register(new DnsNameResolverProvider())

    // Create the service implementation
    val serviceImpl: BedrockLambdaService = new BedrockLambdaServiceImpl()

    val builder: NettyServerBuilder = NettyServerBuilder.forPort(testPort)
    val server = builder.addService(BedrockLambdaService.bindService(serviceImpl, ec)).build().start()

    println(s"gRPC server started, listening on port ${server.getPort}")

    // Add a shutdown hook to gracefully stop the server
    sys.addShutdownHook {
      println("Shutting down gRPC server...")
      server.shutdown()
    }

    // Keep the server alive until it is terminated
    server.awaitTermination()
  }
}
