ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.15"

lazy val root = (project in file("."))
  .settings(
    name := "AkkaServer",
    assembly / assemblyJarName := "AkkaServer.jar",
  )

resolvers += "Akka library repository".at("https://repo.akka.io/maven")
enablePlugins(AkkaGrpcPlugin)

libraryDependencies += "org.scalaj" %% "scalaj-http" % "2.4.2"

// Serialization
libraryDependencies += "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.18.1"

// Akka
val akkaVersion = "2.10.0"
val akkaHttpVersion = sys.props.getOrElse("akka-http.version", "10.7.0")
libraryDependencies ++= Seq(
  "com.typesafe.akka" %% "akka-actor" % akkaVersion,
  "com.typesafe.akka" %% "akka-actor-testkit-typed" % akkaVersion % Test,
  "com.typesafe.akka" %% "akka-http"                % akkaHttpVersion,
  "com.typesafe.akka" %% "akka-http-spray-json"     % akkaHttpVersion,
  "com.typesafe.akka" %% "akka-actor-typed"         % akkaVersion,
  "com.typesafe.akka" %% "akka-stream"              % akkaVersion,
)

val grpcVersion = "1.62.2" // Ensure this is consistent
libraryDependencies ++= Seq(
  "io.grpc" % "grpc-netty" % grpcVersion,
  "io.grpc" % "grpc-core" % grpcVersion,
  "io.grpc" % "grpc-stub" % grpcVersion,
  "io.grpc" % "grpc-protobuf" % grpcVersion,
  "com.thesamet.scalapb" %% "scalapb-runtime-grpc" % scalapb.compiler.Version.scalapbVersion,
  "com.lightbend.akka.grpc" %% "akka-grpc-runtime" % "2.5.0"
)

dependencyOverrides ++= Seq(
  "io.grpc" % "grpc-netty" % grpcVersion,
  "io.grpc" % "grpc-core" % grpcVersion,
  "io.grpc" % "grpc-stub" % grpcVersion,
  "io.grpc" % "grpc-protobuf" % grpcVersion,
  "io.grpc" % "grpc-util" % grpcVersion
)

// gRPC
Compile / PB.targets := Seq(
  scalapb.gen() -> (Compile / sourceManaged).value / "scalapb"
)
Compile / akkaGrpcGeneratedLanguages := Seq(AkkaGrpc.Scala)
Compile / akkaGrpcGeneratedSources := Seq(AkkaGrpc.Client, AkkaGrpc.Server)

// Logging, tests
libraryDependencies ++= Seq(
  "ch.qos.logback" % "logback-classic" % "1.4.11", // Logback Classic for SLF4J 2.x
  "org.slf4j" % "slf4j-api" % "2.0.9",            // SLF4J API 2.x
  "com.typesafe" % "config" % "1.4.3",            // Configuration library
  "org.scalatest" %% "scalatest" % "3.2.19" % Test, // For tests
  "com.typesafe.akka" %% "akka-slf4j" % "2.8.6"   // Akka's SLF4J integration
)


assemblyMergeStrategy in assembly := {
  case PathList("META-INF", "services", xs @ _*) => MergeStrategy.concat
  case PathList("META-INF", _ @ _*)             => MergeStrategy.discard
  case "reference.conf"                         => MergeStrategy.concat
  case x                                        => MergeStrategy.first
}