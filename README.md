# Invoking a Bedrock Lambda with RESTful gRPC Proxy

See the Lambda code over at https://github.com/dbrun3/BedrockLambdaHandler

### To Use

Run the main server and provide a url for your lambda function. You can use mine at https://gvjv3rsq90.execute-api.us-east-1.amazonaws.com/default/bedrockLambda.

    sbt run main https://gvjv3rsq90.execute-api.us-east-1.amazonaws.com/default/bedrockLambda

Curl the server with an input message.

    curl -X POST http://localhost:8080/chat -H "Content-Type: application/json"   -d '{"input":"Who are you?"}'

The server package the input into an `InputMessage` protobuf object where it is serialized and sent 
to the AWS Gateway as a POST request. The Lambda function deserializes it back into the original `InputMessage`
and forwards the input to Amazon Bedrock. The response is packaged as an `OutputMessage` and re-serialized to send
back to our server where it is processed, deserialized and returned to the user. Since Lambda functions are
serverless by design.

    {"response": " I am a chatbot, a computer program designed to engage in conversations with 
    users using text or voice. I am powered by a large language model, and have been trained to 
    help users by providing thorough responses to their questions and assisting them with a variety 
    of tasks. How can I assist you today? "}

### Local Testing and REAL gRPC

The main server can also invoke a genuine gRPC on a test server which is how it was originally 
implemented before realizing lambda cannot actually be accessed this way through API Gateway.


    sbt run main test
Run the grpc test server    

    sbt run grpc
Curl the main server with an input message

    curl -X POST http://localhost:8080/chat -H "Content-Type: application/json"   -d '{"input":"hi"}'

The main server will package the input into a `InputMessage` protobuf object and call the 
`invokeBedrockLambda` function on the grpc server which will just return
"Processed input" followed by the original input string as an `OutputMessage`. This message is
then relayed back to the user.

    {"response": "Processed input: hi"}

### Test Suite

To make sure everything works correctly before trying it yourself:

    sbt test
