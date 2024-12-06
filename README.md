# Distrubuted Cloud Computing for AI - Fall Semester 2024 

### HW1
HW1 is in the 'hw1' branch. To try it yourself, clone the repository and open in in Intellij. Set your sbt shell to jre version 8/11.
You might be able to go higher, but 23 will for sure throw errors. 

For testing, the last two test cases will require you to set your own input and output directories in the case itself.

For running, sbt run will run the default Driver program which takes "operation" "inputfile" "outputfile" args. Operations are "Word2Vec" "CosineSim" and "Tokenization"

Video report: https://www.youtube.com/watch?v=__O1X9f8VuI (Output contains incorrect frequencies, see "output" folder in hw1 branch with updated, accurate output. Since posting video replaced hardcoded word2vec config w a config file, still experimenting with optimal values even now in preparation for hw2) More detailed documentation provided in hw1 branch

### HW2
HW2 is in the 'hw2' branch. To try it yourself, clone the repository and open in in Intellij. Set your sbt shell to jre version 8/11. Scala version 2.12.18 (aready specified in build.sbt)

For testing, the some test cases will require you to set your own filepaths in the case itself.

For running, sbt run will run the default SparkLLMTrain program which takes "input_path" "embedding_path" "model_output_file" "config_file" args.

There is also a TransformerModel program with the main method commented out. This program simply attempts to generate a sentence using the model like in the sample code, and
does not require any arguments. If the output looks terrible, don't blame the program, the model just sucks.

Video report: https://youtu.be/ss_7mzfHIJc

### HW3
HW3 main server implementation is in the 'hw3' branch. To try it yourself, clone the repository and open in in Intellij. Set your sbt shell to jre version 11.
You can test to make sure the configurations are properly set with `sbt test`.

Run the server with `sbt run <server, main or test> <url (lambda endpoint)>`. More details in the branch's README.

The lambda handler code can be found in this repository: https://github.com/dbrun3/BedrockLambdaHandler

Video report: https://youtu.be/yZJUen-e6ME
