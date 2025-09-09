# In code/agent/llm_api.py
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# Use the model ID for the Qwen2.5-0.5B-Instruct model
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading local LLM: {model_id}...")

# Load the tokenizer and model from Hugging Face
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # We will load the model without quantization to simplify troubleshooting.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model from Hugging Face: {e}")
    print("This might be due to insufficient VRAM or a missing dependency.")
    tokenizer = None
    model = None


def call_llm(prompt: str) -> tuple[str, dict]:
    """
    Directly calls a local LLM loaded via Hugging Face Transformers.
    """
    if model is None or tokenizer is None:
        print("Model or tokenizer not loaded. Returning empty response.")
        return "", {"prompt_tokens": 0, "completion_tokens": 0}

    try:
        start_time = time.time()

        messages = [{"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7
        )

        response_text = tokenizer.decode(generated_ids[0])
        response_text = response_text.replace(text, "")

        end_time = time.time()

        prompt_tokens = model_inputs.input_ids.shape[1]
        completion_tokens = generated_ids.shape[1] - prompt_tokens

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": generated_ids.shape[1],
            "latency_ms": (end_time - start_time) * 1000
        }

        return response_text, usage

    except Exception as e:
        print(f"An unexpected error occurred during local inference: {e}")
        return "", {"prompt_tokens": 0, "completion_tokens": 0}