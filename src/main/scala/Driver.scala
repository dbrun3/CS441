object Driver:
  def main(args: Array[String]): Unit =
    if (args.length < 3) {
      println("Usage: Driver <ClassName> <Input> <Output>")
      println(args.length)
      sys.exit(1) // Exits the program with a non-zero status to indicate an error
    }

    val className = args(0)
    val inputPath = args(1)
    val outputPath = args(2)
    
    

    if(className == "Word2Vec") {
      if(args.length < 4) {
        println("Usage: Word2Vec <ClassName> <Input> <Output> <Conf>")
        println(args.length)
      }
      Word2VecMR.run(inputPath, outputPath, args(3))
    }
    else if(className == "CosineSim") {
      CosineSimMR.run(inputPath, outputPath)
    }
    else if(className == "Token") {
      TokenizeMR.run(inputPath, outputPath)
    }
    else {
      println("Unknown class")
      sys.exit(1)
    }

