#!/bin/bash

# Check if required inputs are provided
if [ $# -ne 3 ]; then
  echo "Error: Missing required parameters"
  echo "Usage: ./fine_tune.sh <OPENAI_API_KEY> <FINE_TUNE_JOB_ID> <OUTPUT_FILE>"
  exit 1
fi

# Read input variables
OPENAI_API_KEY=$1
FINE_TUNE_JOB_ID=$2
OUTPUT_FILE=$3

curl https://api.openai.com/v1/files/$FINE_TUNE_JOB_ID/content \
  -H "Authorization: Bearer $OPENAI_API_KEY" > "${OUTPUT_FILE}"
