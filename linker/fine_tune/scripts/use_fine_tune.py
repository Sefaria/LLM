import sys
import json
import requests
import os

# Check if required inputs are provided
if len(sys.argv) != 5:
    print("Error: Missing required parameters")
    print("Usage: python fine_tune.py <OPENAI_API_KEY> <FINE_TUNE_JOB_JSON_FILE> <INPUT_TEXT_FILE> <OUTPUT_FILE>")
    sys.exit(1)

# Read input variables
OPENAI_API_KEY = sys.argv[1]
FINE_TUNE_JOB_JSON_FILE = sys.argv[2]
INPUT_TEXT_FILE = sys.argv[3]
OUTPUT_FILE = sys.argv[4]

PROMPT_STOP_SEQUENCE = "\n\n###\n\n"
COMPLETION_STOP_SEQUENCE = "###"

# Extract model name
with open(FINE_TUNE_JOB_JSON_FILE, 'r') as json_file:
    data = json.load(json_file)
    MODEL_NAME = data['fine_tuned_model']

# Read text file
with open(INPUT_TEXT_FILE, 'r') as file:
    PROMPT = file.read()
    ESCAPED_PROMPT = json.dumps(PROMPT)

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {OPENAI_API_KEY}',
}

payload = {
    'prompt': ESCAPED_PROMPT + PROMPT_STOP_SEQUENCE,
    'max_tokens': 1000,
    'model': MODEL_NAME,
    'stop': COMPLETION_STOP_SEQUENCE,
    'temperature': 0
}

response = requests.post('https://api.openai.com/v1/completions', headers=headers, data=json.dumps(payload))

with open(OUTPUT_FILE, 'w') as outfile:
    json.dump(response.json(), outfile)

with open(OUTPUT_FILE, 'r') as result:
    print(result.read())