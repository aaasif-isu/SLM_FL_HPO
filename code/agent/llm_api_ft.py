# In code/agent/llm_api.py
import os
import time
import json
import re
import torch
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# -----------------------------
# Config
# -----------------------------
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Prefer deterministic outputs for HP JSON (no sampling)
GEN_KW = dict(
    max_new_tokens=256,
    do_sample=False,
    # If you later want sampling, set do_sample=True and add temperature/top_p/top_k.
)

# Optional: 4-bit inference (set to True if you need to save VRAM and have bitsandbytes)
USE_4BIT = False

# -----------------------------
# LoRA discovery
# -----------------------------
def _has_adapter(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    cfg = os.path.join(path, "adapter_config.json")
    safet = os.path.join(path, "adapter_model.safetensors")
    binf  = os.path.join(path, "adapter_model.bin")
    return os.path.isfile(cfg) and (os.path.isfile(safet) or os.path.isfile(binf))

def _pick_lora_dir() -> str | None:
    # 1) env var override
    env_dir = os.getenv("LORA_DIR")
    if _has_adapter(env_dir):
        print(f"Found LoRA adapter via $LORA_DIR: {env_dir}")
        return env_dir

    # 2) ../finetune/qwen_sft_lora (relative to this file)
    agent_dir = os.path.dirname(os.path.abspath(__file__))            # .../code/agent
    finetune_dir = os.path.normpath(os.path.join(agent_dir, "..", "finetune"))
    rel_dir = os.path.join(finetune_dir, "qwen_sft_lora")
    if _has_adapter(rel_dir):
        print(f"Found LoRA adapter next to repo: {rel_dir}")
        return rel_dir

    # 3) absolute path you trained earlier
    abs_dir = "/u/aalasif/SLM_FL_HPO/code/finetune/qwen_sft_lora"
    if _has_adapter(abs_dir):
        print(f"Found LoRA adapter at absolute path: {abs_dir}")
        return abs_dir

    return None

# -----------------------------
# Model / Tokenizer loading
# -----------------------------
print(f"Loading base model: {BASE_MODEL_ID}")

try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Safer for causal LM padding
    tokenizer.padding_side = "right"
except Exception as e:
    print(f"Tokenizer load error: {e}")
    tokenizer = None

try:
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        bnb_config = None

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
    )
except Exception as e:
    print(f"Base model load error: {e}")
    base_model = None

model = None
if base_model is not None:
    lora_dir = _pick_lora_dir()
    if lora_dir:
        try:
            print(f"Attaching LoRA adapter from: {lora_dir}")
            model = PeftModel.from_pretrained(base_model, lora_dir)
            print("LoRA attached.")
        except Exception as e:
            print(f"Failed to load LoRA adapter, using base only. Error: {e}")
            model = base_model
    else:
        print("LoRA folder not found; using base model only.")
        model = base_model
else:
    print("No model loaded.")
    model = None

# -----------------------------
# Helpers
# -----------------------------
_JSON_BLOCK = re.compile(r"\{.*\}", flags=re.S)

def _extract_first_json(s: str) -> str:
    """
    Best-effort: strip ``` fences if present and extract first JSON object.
    Returns original string if JSON not found (caller can still use raw).
    """
    # Remove fenced code block markers (```json ... ```)
    s = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", s, flags=re.IGNORECASE)
    m = _JSON_BLOCK.search(s)
    return m.group(0).strip() if m else s.strip()

# -----------------------------
# Public API
# -----------------------------
def call_llm(prompt: str) -> Tuple[str, dict]:
    """
    Call the (fine-tuned) local LLM.
    `prompt` is your user text (string). We wrap it in a system+user chat and generate.
    Returns (response_text, usage_dict).
    """
    if model is None or tokenizer is None:
        return "", {"prompt_tokens": 0, "completion_tokens": 0}

    try:
        system_msg = (
            "You are a hyperparameter assistant. "
            "Output ONLY JSON with keys {lr,batch,wd,optimizer,scheduler,dropout}."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]

        # Build chat prompt with Qwen template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tokenize and move to device
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        start = time.time()
        gen = model.generate(
            **inputs,
            eos_token_id=tokenizer.eos_token_id,
            **GEN_KW,
        )
        end = time.time()

        # Decode only newly generated tokens (exclude prompt)
        new_tokens = gen[0][inputs.input_ids.shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Optional: try to extract JSON (won't throw if not JSON)
        cleaned = _extract_first_json(decoded)

        usage = {
            "prompt_tokens": int(inputs.input_ids.shape[1]),
            "completion_tokens": int(new_tokens.shape[0]),
            "total_tokens": int(gen.shape[1]),
            "latency_ms": (end - start) * 1000.0,
        }
        return cleaned, usage

    except Exception as e:
        print(f"Local inference error: {e}")
        return "", {"prompt_tokens": 0, "completion_tokens": 0}
