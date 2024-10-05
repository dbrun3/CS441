# CS441_Fall2024
Class repository for CS441 on Cloud Computing taught at the University of Illinois, Chicago in Fall, 2024

## Detailed Documentation

This program leverages Apache Hadoop's MapReduce framework to compute semantic word pairs by generating word embeddings using Word2Vec and comparing them with cosine similarity. The input text is processed in two stages: first, in the Mapper, the text is tokenized, and a Word2Vec model is applied to generate embeddings for each word. The reducer than aggregates the result of each each vocabulary's embeddings. In the second step, the program calculates the cosine similarity between the embeddings of each word and every other word to determine their semantic proximity. Words with the highest similarity scores are considered the most related, and the program outputs word pairs along with their cosine similarity, effectively identifying words with similar meanings based on their vector representations in the text. This approach enables the program to analyze large-scale textual data and extract meaningful word relationships using distributed computing.

### Word2Vec.scala

This file contains a MapReduce implementation using Hadoop and Word2Vec for processing large-scale text data and generating word embeddings. The program reads input text, encodes it using Byte Pair Encoding (BPE), and processes it to generate word embeddings using the Word2Vec model. It is divided into four main components: helper functions, the Mapper, the Reducer, and the driver code for the MapReduce job.

Imports

    com.knuddels.jtokkit: Used for handling token encodings with Byte Pair Encoding (BPE).
    org.apache.hadoop: Provides the MapReduce framework and file system operations.
    org.deeplearning4j.models.word2vec: Provides Word2Vec models for generating word embeddings.
    scala.jdk.CollectionConverters._: Converts Java collections into Scala collections.

Helper Functions and Classes

getEncodingType(encoding: String): EncodingType
    
Purpose: Maps a string to an appropriate EncodingType for Byte Pair Encoding.
    Parameters:
        encoding: String representing the encoding type.
    Returns: Corresponding EncodingType (e.g., CL100K_BASE, R50K_BASE).
    Throws: IllegalArgumentException if the encoding type is unknown.

splitArrayOnToken(arr: Array[Int], separator: Int): List[Array[Int]]

Purpose: Splits an encoded array on a specific token (e.g., a period represented by a token).
    Parameters:
        arr: Array of encoded integers (tokens).
        separator: Token value to split the array on (e.g., period).
    Returns: List of token arrays, each representing a sentence.

TokenizedSequenceIterator

Purpose: Custom iterator that processes a sequence of token arrays and converts them into VocabWord sequences for Word2Vec processing.
    Constructor:
        tokensList: A list of token arrays where each array represents a sequence of tokens (e.g., a sentence).
    Implements: SequenceIterator[VocabWord] interface.
    Methods:
        hasMoreSequences: Checks if more sequences are available.
        nextSequence: Converts the current token array to a sequence of VocabWord.
        reset: Resets the iterator to the beginning.

Mapper Class: W2VMapper

Purpose: Processes input text data, encodes the text using BPE, splits the encoded text into sentences, and generates word embeddings using Word2Vec.

    Extends: Mapper[LongWritable, Text, Text, Text]

Key Functionality:
        Configuration: Reads the configuration file (passed via context) to set up parameters for Word2Vec.
        Text Encoding: Encodes input text into tokens using a specified BPE encoding.
        Token Splitting: Splits the encoded text into sentences based on tokenized periods.
        Word2Vec Model Training: Trains a Word2Vec model on the token sequences.
        Output: Emits the token ID and its corresponding word vector as key-value pairs.

Reducer Class: W2VReducer

Purpose: Aggregates and averages the word vectors (embeddings) emitted by the Mapper, computing the average vector for each token.

    Extends: Reducer[Text, Text, Text, Text]
Key Functionality:
        Aggregation: Sums the embedding vectors for each token emitted by the Mapper.
        Averaging: Computes the average embedding vector for each token by dividing the sum by the count.
        Output: Outputs the token along with its average embedding vector.

Driver Object: Word2VecMR

Purpose: Manages the configuration and execution of the Word2Vec MapReduce job.
    Methods:
        run(input: String, output: String, confFilePath: String): Configures the Hadoop MapReduce job and sets up the Mapper and Reducer classes.
    Key Functionality:
        Job Configuration: Sets up logging, reads the configuration file path, and configures the job with Mapper and Reducer classes.
        Job Execution: Submits the job, waits for it to complete, and handles job success or failure.
        Logging: Uses SLF4J logging to track the job's progress and result.

Overall Flow

Text Input: The input text is processed line-by-line in the W2VMapper.
Encoding: Each line is encoded into tokens using Byte Pair Encoding (BPE).
Sentence Splitting: The encoded tokens are split into sentences using a period as the separator.
Word2Vec Training: The token sequences are passed to the Word2Vec model, which generates word embeddings.
Mapper Output: The Mapper emits each token and its corresponding word vector.
Reducer Aggregation: The Reducer collects word vectors for each token and computes the average vector.
Final Output: The final output is a token and its averaged word embedding vector, representing the semantic meaning of the word.

Dependencies

Hadoop: Provides the MapReduce framework and handles distributed data processing.
Deeplearning4j: Used for the Word2Vec implementation.
JTokKit: Handles the Byte Pair Encoding (BPE) of text data.

How to Run

Ensure that Hadoop is installed and properly configured.
Place the input text data in the appropriate Hadoop input path.
Run the MapReduce job using the driver object Word2VecMR.
Monitor the logs for success or failure of the job.
The output will be written to the specified output path in HDFS.
    
### CosineSim.scala

This file contains a MapReduce program designed to compute cosine similarities between word embeddings. The program processes input text containing words and their corresponding embedding vectors, calculates the cosine similarity between each word’s embedding and every other word's embedding, and outputs the word with its closest neighboring word based on cosine similarity. The program is structured into three main components: the Mapper, the Reducer, and the driver code.

Imports

    org.apache.hadoop.mapreduce.*: Provides the MapReduce framework for distributed data processing.
    org.apache.hadoop.fs.Path: Handles file paths within the Hadoop Distributed File System (HDFS).
    org.apache.hadoop.conf.Configuration: Manages configuration settings for the Hadoop job.
    org.apache.hadoop.io.*: Provides input and output types such as LongWritable, Text used in the MapReduce job.
    scala.jdk.CollectionConverters._: Converts Java collections into Scala collections.
    org.slf4j.LoggerFactory: Provides logging functionality for the MapReduce job.

Mapper Class: WordEmbeddingMapper

Purpose: Processes input text, extracting words and their corresponding embedding vectors, and emits them as key-value pairs.

    Extends: Mapper[LongWritable, Text, Text, Text]
Key Functionality:
        Input Format: Each line of input contains a word and its associated embedding vector in tab-separated format.
        Processing:
            Extracts the word and the embedding vector.
            The key is set as "ALL" for every record.
            The value is the word concatenated with its embedding in a tab-separated format.
        Output:
            Emits the key-value pair where the key is "ALL" and the value is the word and its embedding vector.

Example Input:

    word: This\t1234\t1.0\t[-0.1, 0.2, -0.3]

Example Output:

    Key: "ALL"
    Value: "This\t-0.1, 0.2, -0.3"

Reducer Class: CosineSimilarityReducer

Purpose: Gathers all words and their embeddings emitted by the Mapper and computes the cosine similarity between each word’s embedding and every other word’s embedding to find the closest neighbor.

    Extends: Reducer[Text, Text, Text, Text]
Key Functionality:
        Input: Receives all the words and embeddings as emitted by the Mapper.
        Processing:
            Iterates over the embeddings for each word.
            Calculates the cosine similarity between the current word and every other word using the cosineSimilarity function.
            Determines the word with the highest similarity score.
        Output:
            Emits the current word, its closest neighboring word, and the cosine similarity score as key-value pairs.

Cosine Similarity Calculation:

    Formula:
    similarity=∑(ai×bi)∑(ai2)×∑(bi2)
    similarity=∑(ai2​)
    ×∑(bi2​)

    ​∑(ai​×bi​)​ where a and b are the embedding vectors of two words.

Example Output:

    Key: "Closest to This: That"
    Value: "Similarity: 0.95"

Helper Method:

    cosineSimilarity(vecA: Array[Double], vecB: Array[Double]): Double:
Computes the cosine similarity between two embedding vectors.
        Takes two arrays of doubles (vectors) as input.
        Returns the cosine similarity score.

Driver Object: CosineSimMR

Purpose: Configures and runs the MapReduce job to compute cosine similarities between word embeddings.
    Methods:
        run(input: String, output: String):
            Configures the Hadoop job, specifying the Mapper, Reducer, input/output formats, and logging levels.
            Submits the job and waits for its completion.
    Key Functionality:
        Job Configuration: Sets the Mapper class (WordEmbeddingMapper), Reducer class (CosineSimilarityReducer), and input/output types (Text for both key and value).
        Logging: Uses SLF4J to log the job's progress and completion.
        Job Execution: Submits the job and waits for it to finish, logging whether it succeeded or failed.

Overall Flow

Input: Text file where each line contains a word and its embedding vector, e.g., word: This\t1234\t1.0\t[-0.1, 0.2, -0.3].
    Mapper Processing:
        The WordEmbeddingMapper extracts the word and its embedding.
        Emits a key-value pair where the key is "ALL" and the value is the word and its embedding vector.
    Reducer Processing:
        The CosineSimilarityReducer collects the embeddings, calculates the cosine similarity between each word and every other word, and identifies the closest neighbor.
        Emits a key-value pair where the key is the word and its closest neighbor, and the value is their cosine similarity score.
    Output: The final output is a list of words, each paired with its closest neighbor and the similarity score.

How to Run

Ensure that Hadoop is properly configured and that the input data is available in the Hadoop Distributed File System (HDFS).
Run the CosineSimMR.run(inputPath, outputPath) method, where inputPath is the location of the input file and outputPath is where the output will be written.
Monitor the logs for job progress and output once the job is complete.
