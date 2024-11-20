addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "2.2.0")

resolvers += "Akka library repository".at("https://repo.akka.io/maven")
addSbtPlugin("com.lightbend.akka.grpc" % "sbt-akka-grpc" % "2.5.0")