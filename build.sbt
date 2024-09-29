ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.5.0"

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

// https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-common
libraryDependencies += "org.apache.hadoop" % "hadoop-common" % "3.3.4"
// https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-mapreduce-client-core
libraryDependencies += "org.apache.hadoop" % "hadoop-mapreduce-client-core" % "3.3.4"
libraryDependencies += "org.apache.hadoop" % "hadoop-mapreduce-client-jobclient" % "3.3.4"

libraryDependencies += "com.knuddels" % "jtokkit" % "1.1.0" // Adjust version as needed

libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M1.1"
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-M1.1"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "1.0.0-M1.1"

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
