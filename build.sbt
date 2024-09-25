ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.5.0"

lazy val root = (project in file("."))
  .settings(
    name := "Exercises441"
  )

// https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-common
libraryDependencies += "org.apache.hadoop" % "hadoop-common" % "3.3.4"
// https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-mapreduce-client-core
libraryDependencies += "org.apache.hadoop" % "hadoop-mapreduce-client-core" % "3.3.4"
libraryDependencies += "org.apache.hadoop" % "hadoop-mapreduce-client-jobclient" % "3.3.4"

libraryDependencies += "com.knuddels" % "jtokkit" % "1.1.0" // Adjust version as needed

libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta7"
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-M1.1"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta7"