# SparkLLMTraining Documentation

The SparkLLMTraining object is a Scala program designed to train a language model using Apache Spark for distributed processing. This document explains each section of the code and provides a detailed walkthrough of the main function.
Imports and Setup

The program uses the following key libraries:

* Apache Spark for distributed data processing. 
* DL4J (Deeplearning4j) for deep learning with a multi-layer network on Spark.
* Config to load configuration parameters for model training.
* SLF4J for logging.

### Step-by-Step Walkthrough

### 1. Argument Parsing:
The program expects four arguments: input data path, model output path, embeddings path, and configuration file path.
If the arguments are not provided, it logs an error and terminates.

### 2. Initialize Spark Context:
Creates the Spark context in local mode for testing ("local[*]"). This mode can be modified to run on a cluster in production.

### 3. Load Configuration:
Loads the model configuration from the specified configuration file. Key parameters like averaging frequency, batch size, and learning rate are retrieved for setting up the model and training master.

### 4. Load Embeddings:
The program loads precomputed embeddings from a file into an embedding map using the Spark context. The embedding map is then broadcasted across Spark nodes to ensure every worker can access it efficiently.

### 5. Define Model Parameters:
Retrieves model-specific configurations, such as embeddingDim, vocabSize, hiddenSize, learningRate, batchSize, windowSize, and numEpochs. These define the architecture and training parameters for the model.

### 6. Load and Process Sentences:
Reads sentences from the input file into an RDD. Each sentence will be divided into sliding windows to provide context for the language model.
Logs the total number of sentences loaded.

### 7. Create Sliding Windows:
Creates sliding windows for each sentence in the RDD using a helper function, createSlidingWindowsWithPositionalEmbedding. Each window represents a segment of the sentence with positional embeddings.
Caches the slidingWindowsRDD to optimize performance since it will be reused in later steps.

    // Updated function to use an existing RDD instead of creating a new one
    def createSlidingWindowsWithPositionalEmbedding(tokenString: String, windowSize: Int, vocabSize: Int, embeddingDim: Int, embeddingMap: Broadcast[Map[Int, INDArray]]): Iterator[DataSet] = {

        // Encode tokens based on the broadcasted encoding
        val tokens: Array[Int] = encodeTokens(tokenString)
    
        if (tokens.length <= windowSize) return Iterator.empty
    
        // Use sliding to create windows directly
        tokens.sliding(windowSize + 1).map { window: Array[Int] =>
    
            // Extract input window (first `windowSize` tokens) and target token (last token)
            val inputWindow = window.take(windowSize)
            
            // Use broadcasted embedding map to embed the tokens
            val features: INDArray = tokenizeAndEmbed(inputWindow, embeddingMap.value, embeddingDim)
            
            // Initialize the one-hot encoded target vector using the broadcast vocab size
            val label = Nd4j.zeros(1, vocabSize, windowSize)
            
            // Set target index for each timestep
            for (t <- 0 until windowSize) {
            label.putScalar(Array(0, window(t + 1), t), 1.0)
            }
    
          new DataSet(features, label)
        }
    }

### 8. Batch the Sliding Windows:
Batches the sliding windows by partition. The batchSlidingWindows function groups windows into batches based on batchSize, making them ready for distributed training.
Logs the total number of batches created.
    
    def batchSlidingWindows(iter: Iterator[DataSet], batchSize: Int, embeddingDim: Int, vocabSize: Int): Iterator[DataSet] = {
        val batchedList = ArrayBuffer[DataSet]()
        val batchBuffer = ArrayBuffer[DataSet]()
    
        iter.foreach { dataSet =>
            // Add each DataSet to the current batch buffer
            batchBuffer += dataSet
            
            if (batchBuffer.size == batchSize) {
            val batchDataSet = DataSet.merge(batchBuffer.asJava)
            val originalShape = batchDataSet.getFeatures.shape()
            val windowSize: Int = (originalShape(0) / batchSize).toInt
            
            val reshapedFeatures = batchDataSet.getFeatures.reshape(windowSize, batchSize, embeddingDim).permute(1, 2, 0)
            val reshapedLabels = batchDataSet.getLabels.reshape(batchSize, vocabSize, windowSize)
            
            batchDataSet.setFeatures(reshapedFeatures)
            batchDataSet.setLabels(reshapedLabels)
            
            batchedList += batchDataSet
            batchBuffer.clear()
            }
        }
        batchedList.iterator
    }    

### 9. Initialize and Configure Model:
Creates a deep learning model using the specified dimensions (embeddingDim, hiddenSize, vocabSize, learningRate).
Sets up the TrainingMaster configuration with parameters such as averagingFrequency and workerPrefetchNumBatches, which control how model parameters are synchronized across workers.

### 10. Create Distributed Model:
Initializes a SparkDl4jMultiLayer model using the Spark context and the previously created neural network model. This enables distributed training across Spark nodes.

### 11. Add Training Listeners:
Adds a ScoreIterationListener to monitor the model's training score every 10 iterations, helping with performance tracking and debugging.

### 12. Train the Model:
Loops over the specified number of epochs. For each epoch:
Records the start time and logs the epoch number.
Trains the model by calling fit on the batchedWindowsRDD.
Retrieves and logs the model's score for each epoch. 
Calculates and logs the duration for each epoch.

### 13. Save Trained Model:
After training, saves the model to the specified output path.

## Results and Limitations

The following is from a local test of the model. Parameters and input size were selected for 
maximizing training speed since many epochs would be required to fit the data. Even after 200 epochs,
the results show that while it may have learned the vocabulary of the set, this is not nearly enough training to
for the model to truly memorize the set or make any meaningful generalizations. This was also reflected in the results
of the model trained on the distributed network on a larger training set. With 200 epochs over a larger input taking several hours
to compute, the time and cost for training that many epochs simply was not worth any meaningful outcome.

Input:

    In this northern land they had always poopy fart placed their altars on the top of a mountain, to be close to heaven.
    In this northern land they had always placed their altars on the top of a mountain, to be close to heaven.
    In this northern land they had always placed their altars on the top of a mountain.
    He is.
    They had always placed their altars on the top of a mountain.

Test model config:

    model {
        hiddenSize = 256         // Increase hidden size for more learning capacity
        learnRate = 0.001        // Slightly higher learning rate to improve learning
        batchSize = 2            // Small batch size for small input
        windowSize = 4           // Short input sentences
        numEpochs = 200          // Allow more epochs to let the model learn more from the limited data
    }

Post training test:

    Sequence 0 - Label max index: 484, Top 5 Prediction indices: 262, 1353, 319, 945, 9538
    Sequence 0 - Label (translated):  they
    Prediction index: 262 (translated:  the), Score: 0.5093519687652588
    Prediction index: 1353 (translated:  top), Score: 0.3155979812145233
    Prediction index: 319 (translated:  on), Score: 0.12673380970954895
    Prediction index: 945 (translated: ars), Score: 0.020852144807577133
    Prediction index: 9538 (translated:  heaven), Score: 0.004798715468496084
    Sequence 1 - Label max index: 550, Top 5 Prediction indices: 262, 1353, 319, 945, 9538
    Sequence 1 - Label (translated):  had
    Prediction index: 262 (translated:  the), Score: 0.50911545753479
    Prediction index: 1353 (translated:  top), Score: 0.31792736053466797
    Prediction index: 319 (translated:  on), Score: 0.1253281831741333
    Prediction index: 945 (translated: ars), Score: 0.020568309351801872
    Prediction index: 9538 (translated:  heaven), Score: 0.0047335135750472546
