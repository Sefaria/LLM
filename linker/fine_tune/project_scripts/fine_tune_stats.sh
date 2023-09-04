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
RESULTS_FILE_ID=$(jq -r '.id' $FINE_TUNE_JOB_JSON_FILE)

RESULTS_FILES_LENGTH=$(jq -r '.result_files | length' $FINE_TUNE_JOB_JSON_FILE)

if [ $RESULTS_FILES_LENGTH -eq 0 ]; then
  echo "No results yet."
  exit 1
fi

RESULTS_FILE_ID=$(jq -r '.result_files[0].id' $FINE_TUNE_JOB_JSON_FILE)

curl https://api.openai.com/v1/files/$RESULTS_FILE_ID/content \
  -H "Authorization: Bearer $OPENAI_API_KEY" > "${OUTPUT_FILE}"
