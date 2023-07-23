#!/bin/bash

# Check if required inputs are provided
if [ $# -ne 4 ]; then
  echo "Error: Missing required parameters"
  echo "Usage: ./fine_tune.sh <OPENAI_API_KEY> <FINE_TUNE_JOB_JSON_FILE> <INPUT_TEXT_FILE> <OUTPUT_FILE>"
  exit 1
fi

# Read input variables
OPENAI_API_KEY=$1
FINE_TUNE_JOB_JSON_FILE=$2
INPUT_TEXT_FILE=$3
OUTPUT_FILE=$4

PROMPT_STOP_SEQUENCE="\n\n###\n\n"
COMPLETION_STOP_SEQUENCE=" \$\$END#INDICATOR\$\$"

# Check if jq is installed
if ! command -v jq &>/dev/null; then
  echo "Error: jq command not found. Please install jq package."
  exit 1
fi

# Extract model name
MODEL_NAME=$(jq -r '.fine_tuned_model' $FINE_TUNE_JOB_JSON_FILE)

PROMPT=$(cat "$INPUT_TEXT_FILE")

curl https://api.openai.com/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d "{
    \"prompt\": \"$PROMPT$PROMPT_STOP_SEQUENCE\",
    \"max_tokens\": 1000,
    \"model\": \"$MODEL_NAME\",
    \"stop\": \"$COMPLETION_STOP_SEQUENCE\",
    \"temperature\": 0
  }" > "$OUTPUT_FILE"

cat $OUTPUT_FILE
