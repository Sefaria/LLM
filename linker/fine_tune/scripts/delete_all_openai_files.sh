#!/bin/bash

# Check for required number of arguments
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 OPENAI_API_KEY"
  exit 1
fi

# Assign inputs to variables
OPENAI_API_KEY="$1"

# Store JSON output from listing API in a temporary file
curl https://api.openai.com/v1/files \
  -H "Authorization: Bearer $OPENAI_API_KEY" > tmp.json

# Extract file ids from JSON using the jq command-line tool
# and store them in an array called file_ids
file_ids=($(jq -r '.data[].id' tmp.json))

# Iterate over file ids and delete each file by invoking the delete API
for file_id in "${file_ids[@]}"; do
  curl https://api.openai.com/v1/files/$file_id \
    -X DELETE \
    -H "Authorization: Bearer $OPENAI_API_KEY"
done

# Remove the temporary JSON file
rm tmp.json