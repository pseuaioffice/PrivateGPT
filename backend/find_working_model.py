import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Test a list of models to see which ones work with the Router
models_to_test = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "microsoft/Phi-3-small-8k-instruct",
]

client = InferenceClient(api_key=TOKEN)

for model in models_to_test:
    try:
        response = client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=5
        )
        print(f"[SUCCESS] {model} => {response.choices[0].message.content.strip()}")
        break  # stop at first working model
    except Exception as e:
        error_msg = str(e)
        if "model_not_supported" in error_msg:
            print(f"[NOT SUPPORTED] {model}")
        elif "rate" in error_msg.lower() or "429" in error_msg:
            print(f"[RATE LIMITED]  {model}")
        else:
            print(f"[ERROR]  {model}: {error_msg[:120]}")
