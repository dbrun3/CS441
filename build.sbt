ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "Exercises441",
      assembly / mainClass := Some("Driver")
  )

lazy val utils = (project in file("utils"))
  .settings(
    assembly / assemblyJarName := "Driver.jar",
    // more settings here ...
  )

libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.19" % Test
libraryDependencies += "com.typesafe" % "config" % "1.4.3"

// Add Logback Classic and SLF4J dependencies
libraryDependencies ++= Seq(
  "ch.qos.logback" % "logback-classic" % "1.2.11", // Logback Classic
  "org.slf4j" % "slf4j-api" % "1.7.36"             // SLF4J API
)

libraryDependencies += "com.knuddels" % "jtokkit" % "1.1.0" // Adjust version as needed

libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1"
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-M2.1"
libraryDependencies += "org.deeplearning4j" % "dl4j-spark3_2.12" % "1.0.0-M2"
libraryDependencies += "org.deeplearning4j" % "dl4j-spark-parameterserver_2.12" % "1.0.0-M2.1"

libraryDependencies += "org.apache.spark" % "spark-core_2.12" % "3.5.3"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.12" % "3.5.3"


libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1"

libraryDependencies ++= Seq(
  "org.bytedeco" % "openblas" % "0.3.21-1.5.8" classifier "linux-x86_64",
  "org.bytedeco" % "openblas" % "0.3.21-1.5.8"
)

// Assembly settings to include all dependencies
assembly / assemblyMergeStrategy := {
  case PathList("META-INF", "services", xs @ _*) => MergeStrategy.concat
  case PathList("META-INF", "MANIFEST.MF") => MergeStrategy.discard
  case PathList("META-INF", xs @ _*) => MergeStrategy.concat
  case PathList("native", xs @ _*) => MergeStrategy.first
  case PathList("native-libs", xs @ _*) => MergeStrategy.first // Include native libs
  case PathList("org", "bytedeco", "openblas", xs @ _*) => MergeStrategy.first // Include native OpenBLAS binaries
  case _ => MergeStrategy.first
}
