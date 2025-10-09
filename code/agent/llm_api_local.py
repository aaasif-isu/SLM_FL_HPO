# code/agent/llm_api.py

import time
import json
import re
from typing import Tuple, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Choose your local instruct model
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading local LLM: {model_id}...")

# Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
    )
    # Ensure pad token exists to avoid attention mask warnings
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model from Hugging Face: {e}")
    print("This might be due to insufficient VRAM or a missing dependency.")
    tokenizer = None
    model = None

# ---------- helpers (no change to prompts.py needed) ----------

SYSTEM_JSON_ONLY = (
    "You are a function that returns JSON only. "
    "Output must be a single valid JSON object. "
    "Do not include markdown, backticks, or any text outside JSON."
)

# Lightweight wrapper to discourage repeating the same HPs across rounds
ANTI_STALE_HINT = (
    "Policy:\n"
    "- If this is NOT the first round and prior hyperparameters appear in the context/history, "
    "avoid repeating identical values; make at least one meaningful change within allowed constraints.\n"
    "- If train accuracy is much higher than test accuracy, reduce learning_rate (e.g., x0.5) OR increase weight_decay (e.g., x2), "
    "OR decrease local_epochs by 1 (not below min), OR lower batch_size to the next allowed option; pick exactly one change.\n"
    "- If both train and test accuracies are low, increase learning_rate (e.g., x1.5) within the allowed max.\n"
    "- Never copy example values verbatim. Always personalize based on history/peer signals.\n"
    "- Output must remain a single valid JSON object."
)

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    # Remove leading ```... and trailing ```
    if s.startswith("```"):
        # drop the first line fence (may include language tag)
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline + 1 :]
        else:
            s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

def _first_json_object(s: str):
    """
    Return the first balanced {...} JSON object found in s.
    Handles nested braces and ignores braces inside quoted strings.
    """
    start = -1
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        return s[start : i + 1]
    return None

def _sanitize_to_json(text: str) -> str:
    """
    Trim Qwen terminators, strip code fences, then extract first JSON object.
    If none is found, return the original trimmed text.
    """
    s = text.strip()
    # Trim at Qwen's chat terminator if present
    s = s.split("<|im_end|>", 1)[0].strip()
    s = _strip_code_fences(s)
    obj = _first_json_object(s)
    return obj.strip() if obj else s

# ---------- main call ----------

def call_llm(prompt: str) -> Tuple[str, Dict]:
    """
    Calls the local Qwen Instruct model, enforcing JSON-only output via a system message.
    Sanitizes the reply into a parseable JSON string.
    Returns (json_like_text, usage_dict).
    """
    if model is None or tokenizer is None:
        print("Model or tokenizer not loaded. Returning empty response.")
        return "", {"prompt_tokens": 0, "completion_tokens": 0}

    try:
        start_time = time.time()

        # Wrap your existing prompt with a small anti-stale hint (no change to prompts.py)
        wrapped_prompt = ANTI_STALE_HINT + "\n\n" + prompt

        messages = [
            {"role": "system", "content": SYSTEM_JSON_ONLY + " Prefer adjustments over repetition. Never copy example values."},
            {"role": "user", "content": wrapped_prompt},
        ]

        # Apply Qwen chat template
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize with attention mask & padding
        model_inputs = tokenizer(
            chat_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)

        # Deterministic decoding (no unsupported flags like top_k)
        gen_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,           # deterministic
            temperature=0.0,           # ignored when do_sample=False, safe to leave at 0.0
            top_p=1.0,                 # ignored when do_sample=False
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Slice only the newly generated portion, then decode
        new_tokens = gen_ids[0, model_inputs.input_ids.shape[1]:]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Trim at Qwen end marker and sanitize to first JSON object
        cleaned = _sanitize_to_json(raw_text)

        end_time = time.time()

        usage = {
            "prompt_tokens": int(model_inputs.input_ids.numel()),
            "completion_tokens": int(new_tokens.numel()),
            "total_tokens": int(gen_ids.numel()),
            "latency_ms": (end_time - start_time) * 1000.0,
        }

        return cleaned, usage

    except Exception as e:
        print(f"An unexpected error occurred during local inference: {e}")
        return "", {"prompt_tokens": 0, "completion_tokens": 0}
