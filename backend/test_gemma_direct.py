import os
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MODEL_ID = "google/gemma-2-2b-it"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
headers = {"Authorization": f"Bearer {TOKEN}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

print(f"Testing direct endpoint for: {MODEL_ID}")
output = query({
	"inputs": "<start_of_turn>user\nSay hello!<end_of_turn>\n<start_of_turn>model\n",
    "parameters": {"max_new_tokens": 10}
})
print(f"Output: {output}")
