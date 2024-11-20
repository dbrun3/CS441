ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.15"

lazy val root = (project in file("."))
  .settings(
    name := "AkkaServer" ,
    assembly / assemblyJarName := "AkkaServer.jar",
  )

resolvers += "Akka library repository".at("https://repo.akka.io/maven")
enablePlugins(AkkaGrpcPlugin)


// Serialization
libraryDependencies += "com.fasterxml.jackson.module" % "jackson-module-scala_2.11" % "2.5.2"

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

// Logging, tests, config
libraryDependencies ++= Seq(
  "ch.qos.logback" % "logback-classic" % "1.2.11", // Logback Classic
  "org.slf4j" % "slf4j-api" % "1.7.36", // SLF4J API
  "com.typesafe" % "config" % "1.4.3",
  "org.scalatest" %% "scalatest" % "3.2.19" % Test
)

assemblyMergeStrategy in assembly := {
  {
    case PathList("META-INF", xs@_*) => MergeStrategy.discard
    case x => MergeStrategy.first
  }
}