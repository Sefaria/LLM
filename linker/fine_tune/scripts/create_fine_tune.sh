#!/bin/bash

# Check if required inputs are provided
if [ $# -ne 6 ]; then
  echo "Error: Missing required parameters"
  echo "Usage: ./fine_tune.sh <OPENAI_API_KEY> <BASE_MODEL> <TRAINING_UPLOAD_JSON_FILE> <VALIDATION_UPLOAD_JSON_FILE> <OUTPUT_FILE> <MODEL_SUFFIX>"
  exit 1
fi

# Read input variables
OPENAI_API_KEY=$1
BASE_MODEL=$2
TRAINING_UPLOAD_JSON_FILE=$3
VALIDATION_UPLOAD_JSON_FILE=$4
OUTPUT_FILE=$5
MODEL_SUFFIX=$6

# Check if jq is installed
if ! command -v jq &>/dev/null; then
  echo "Error: jq command not found. Please install jq package."
  exit 1
fi

# Extract file IDs from JSON files
TRAINING_UPLOAD_JSON_FILE_ID=$(jq -r '.id' $TRAINING_UPLOAD_JSON_FILE)
VALIDATION_UPLOAD_JSON_FILE_ID=$(jq -r '.id' $VALIDATION_UPLOAD_JSON_FILE)

# Send cURL request to fine-tune API
curl https://api.openai.com/v1/fine-tunes \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d "{ \
    \"training_file\": \"$TRAINING_UPLOAD_JSON_FILE_ID\", \
    \"validation_file\": \"$VALIDATION_UPLOAD_JSON_FILE_ID\", \
    \"model\": \"$BASE_MODEL\", \
    \"suffix\": \"$MODEL_SUFFIX\" \
  }" > "${OUTPUT_FILE}"

cat $OUTPUT_FILE