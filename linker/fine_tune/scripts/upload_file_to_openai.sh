#!/bin/bash

# Check for required number of arguments
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 OPENAI_API_KEY FILE_TO_UPLOAD OUTPUT_FILE"
  exit 1
fi

# Assign inputs to variables
OPENAI_API_KEY="$1"
FILE_TO_UPLOAD="$2"
OUTPUT_FILE="$3"

# Perform command
curl https://api.openai.com/v1/files \
  -H "Authorization: Bearer ${OPENAI_API_KEY}" \
  -F purpose="fine-tune" \
  -F file="@${FILE_TO_UPLOAD}" > "${OUTPUT_FILE}"
