#!/bin/bash

# Check if required inputs are provided
if [ $# -ne 3 ]; then
  echo "Error: Missing required parameters"
  echo "Usage: ./fine_tune.sh <OPENAI_API_KEY> <RESULTS_FILE_ID> <OUTPUT_FILE>"
  exit 1
fi

# Read input variables
OPENAI_API_KEY=$1
RESULTS_FILE_ID=$2
OUTPUT_FILE=$3

curl https://api.openai.com/v1/files/$RESULTS_FILE_ID/content \
  -H "Authorization: Bearer $OPENAI_API_KEY" > "${OUTPUT_FILE}"
