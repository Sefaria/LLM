#!/bin/bash

# Check if required inputs are provided
if [ $# -ne 3 ]; then
  echo "Error: Missing required parameters"
  echo "Usage: ./fine_tune.sh <OPENAI_API_KEY> <FINE_TUNE_JOB_JSON_FILE> <OUTPUT_FILE>"
  exit 1
fi

# Read input variables
OPENAI_API_KEY=$1
FINE_TUNE_JOB_JSON_FILE=$2
OUTPUT_FILE=$3

# Check if jq is installed
if ! command -v jq &>/dev/null; then
  echo "Error: jq command not found. Please install jq package."
  exit 1
fi

# Extract file IDs from JSON files
FINE_TUNE_ID=$(jq -r '.id' $FINE_TUNE_JOB_JSON_FILE)

curl https://api.openai.com/v1/fine-tunes/$FINE_TUNE_ID \
  -H "Authorization: Bearer $OPENAI_API_KEY" > $OUTPUT_FILE

echo $(cat $OUTPUT_FILE)
