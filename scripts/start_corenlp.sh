#!/bin/bash
# This script starts the Stanford CoreNLP server.
# It assumes the script is run from the project root directory.

CORENLP_DIR="stanford-corenlp-4.5.10"

# Check if the directory exists
if [ ! -d "$CORENLP_DIR" ]; then
  echo "Error: Stanford CoreNLP directory not found at '$CORENLP_DIR'."
  echo "Please ensure you have downloaded and placed it in the project root."
  exit 1
fi

echo "Starting Stanford CoreNLP server..."
# The -cp argument sets the classpath. We need to point it to all the JAR files
# inside the CoreNLP directory. The wildcard '*' does this.
java -mx4g -cp "$CORENLP_DIR/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9002 -timeout 30000
 