import os
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MODEL_ID = "google/gemma-2-2b-it"
# Specific model endpoint
URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "model": MODEL_ID,
    "messages": [{"role": "user", "content": "Say hello!"}],
    "max_tokens": 10
}

print(f"Testing URL: {URL}")
res = requests.post(URL, headers=headers, json=payload)
print(f"Status: {res.status_code}")
print(f"Body: {res.text}")
