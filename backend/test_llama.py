import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

client = InferenceClient(api_key=TOKEN)

print(f"Testing model: {MODEL_ID}")
try:
    response = client.chat_completion(
        model=MODEL_ID,
        messages=[{"role": "user", "content": "Say hello!"}],
        max_tokens=10
    )
    print("SUCCESS!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"FAILURE: {e}")
