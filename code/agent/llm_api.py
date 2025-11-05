# code/agent/llm_api.py
import os
import time
import json
from typing import Tuple, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import requests

from . import shared_state  # reads shared_state.CONFIG

# --- Adapter runtime (unchanged API) ---
from .policy_adapter import (
    init_adapter_runtime,
    get_infer_model,
    set_active_adapter_key,   # re-exported
    policy_update,            # re-exported
)

# ===================== Config & ENV =====================

load_dotenv()

CFG = shared_state.CONFIG or {}

# Backends: "local" | "openrouter" | "auto"
BACKEND = (CFG.get("agents", {}).get("backend") or "local").lower().strip()
SLM_MODEL_ID = CFG.get("agents", {}).get("slm_model", "Qwen/Qwen2.5-0.5B-Instruct")
API_MODEL_ID = CFG.get("agents", {}).get("api_model", "openai/gpt-4o-mini")

# If False (default), we will NOT run local model on CPU.
# In auto mode, this forces fallback to API when no CUDA.
LOCAL_ALLOW_CPU = bool(CFG.get("agents", {}).get("local_allow_cpu", False))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_SITE    = os.getenv("OPENROUTER_SITE", "http://localhost:3000")
OPENROUTER_APP     = os.getenv("OPENROUTER_APP", "FedHPO")

# ===================== Prompts / Helpers (unchanged) =====================

SYSTEM_JSON_ONLY = (
    "You are a strict JSON generator. Return ONLY a single JSON object. "
    "The ONLY keys allowed are 'reasoning', 'hps'. The 'hps' key MUST "
    "contain 'client', 'server', and 'mu' dictionaries/values. "
    "DO NOT use bullet points, hyphens, or lists within the JSON object. "
)

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
    if s.startswith("```"):
        first_newline = s.find("\n")
        s = s[first_newline + 1 :] if first_newline != -1 else s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

def _first_json_object(s: str) -> Optional[str]:
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
    s = text.strip()
    s = s.split("<|im_end|>", 1)[0].strip()   # Qwen end marker, if present
    s = _strip_code_fences(s)
    obj = _first_json_object(s)
    return obj.strip() if obj else s

def _safe_set_pad(tok: AutoTokenizer):
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

# ===================== Local Model (Qwen) =====================

_tokenizer: Optional[AutoTokenizer] = None
_base_model = None
_local_ready = False

def _maybe_init_local() -> None:
    """
    Lazily initialize local tokenizer/model + adapter runtime.
    Respects LOCAL_ALLOW_CPU. If no CUDA and LOCAL_ALLOW_CPU=False, skip init.
    """
    global _tokenizer, _base_model, _local_ready

    if _local_ready:
        return

    cuda_ok = torch.cuda.is_available()
    if not cuda_ok and not LOCAL_ALLOW_CPU:
        print("[llm_api] CUDA not available and agents.local_allow_cpu=False; skipping local init.")
        _tokenizer = None
        _base_model = None
        _local_ready = False
        return

    try:
        print(f"[llm_api] Loading local LLM: {SLM_MODEL_ID} (device: {'cuda' if cuda_ok else 'cpu'}) ...")
        tok = AutoTokenizer.from_pretrained(SLM_MODEL_ID, trust_remote_code=True)
        _safe_set_pad(tok)

        model = AutoModelForCausalLM.from_pretrained(
            SLM_MODEL_ID,
            device_map="auto" if cuda_ok else None,  # if cpu, keep None to .to('cpu') explicitly
            trust_remote_code=True,
        )

        if cuda_ok:
            model = model.eval()  # on GPU via device_map=auto
        else:
            model = model.to("cpu").eval()

        # Build adapter cfg from YAML
        m = CFG.get("model", {})
        l = m.get("lora", {}) or {}
        adapter_cfg = {
            "enabled": bool(m.get("use_lora", False)),
            "adapter_mode": m.get("adapter_mode", "per_cluster"),   # "per_cluster" | "single"
            "lora_r": int(l.get("r", 4)),
            "lora_alpha": int(l.get("alpha", 4)),
            "lora_dropout": float(l.get("dropout", 0.05)),
            "step_lr": float(l.get("step_lr", 2e-5)),
            "max_grad_norm": float(l.get("max_grad_norm", 0.5)),
            "kl_max": float(l.get("kl_max", 0.01)),
            "every_k_rounds": int(l.get("every_k_rounds", 3)),
        }

        init_adapter_runtime(base_model=model, tok=tok, cfg=adapter_cfg)
        _tokenizer = tok
        _base_model = model
        _local_ready = True
        print(f"[llm_api] Local model ready. LoRA enabled={adapter_cfg['enabled']} mode={adapter_cfg['adapter_mode']}")
    except Exception as e:
        print(f"[llm_api] Local init failed: {e}")
        _tokenizer = None
        _base_model = None
        _local_ready = False

def _call_llm_local(prompt: str) -> Tuple[str, Dict]:
    """
    Use local Qwen model (and adapter runtime). Returns (json_text, usage).
    If local not available (e.g., no CUDA and LOCAL_ALLOW_CPU=False), returns ("", usage).
    """
    _maybe_init_local()
    model = get_infer_model()
    tok = _tokenizer

    if model is None or tok is None:
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "backend": "local_unavailable"}

    try:
        start = time.time()
        messages = [
            {"role": "system", "content": SYSTEM_JSON_ONLY + " Prefer adjustments over repetition. Never copy example values."},
            {"role": "user", "content": ANTI_STALE_HINT + "\n\n" + prompt},
        ]
        chat_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        device = next(model.parameters()).device
        inputs = tok(chat_text, return_tensors="pt", padding=True, truncation=True).to(device)

        gen_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        new_tokens = gen_ids[0, inputs.input_ids.shape[1]:]
        raw_text = tok.decode(new_tokens, skip_special_tokens=True)
        cleaned = _sanitize_to_json(raw_text)

        end = time.time()
        usage = {
            "prompt_tokens": int(inputs.input_ids.numel()),
            "completion_tokens": int(new_tokens.numel()),
            "total_tokens": int(gen_ids.numel()),
            "latency_ms": (end - start) * 1000.0,
            "backend": "local",
            "model": SLM_MODEL_ID,
            "device": str(device),
        }
        return cleaned, usage

    except Exception as e:
        print(f"[llm_api] Local inference error: {e}")
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "backend": "local_err", "model": SLM_MODEL_ID}

# ===================== OpenRouter API =====================

def _call_llm_openrouter(prompt: str) -> Tuple[str, Dict]:
    """
    Call OpenRouter API and return (json_text, usage).
    """
    if not OPENROUTER_API_KEY:
        print("[llm_api] OPENROUTER_API_KEY not set.")
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "backend": "openrouter_unavailable"}

    try:
        start = time.time()
        messages = [
            {"role": "system", "content": SYSTEM_JSON_ONLY + " Prefer adjustments over repetition. Never copy example values."},
            {"role": "user", "content": ANTI_STALE_HINT + "\n\n" + prompt},
        ]
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": OPENROUTER_SITE,
            "X-Title": OPENROUTER_APP,
            "Content-Type": "application/json",
        }
        data = {
            "model": API_MODEL_ID,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 512,
        }
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=60,
        )
        resp.raise_for_status()
        j = resp.json()

        content = j.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        cleaned = _sanitize_to_json(content)

        usage_api = j.get("usage", {}) or {}
        end = time.time()
        usage = {
            "prompt_tokens": int(usage_api.get("prompt_tokens", 0)),
            "completion_tokens": int(usage_api.get("completion_tokens", 0)),
            "total_tokens": int(usage_api.get("total_tokens", usage_api.get("prompt_tokens", 0) + usage_api.get("completion_tokens", 0))),
            "latency_ms": (end - start) * 1000.0,
            "backend": "openrouter",
            "model": API_MODEL_ID,
        }
        return cleaned, usage

    except requests.exceptions.HTTPError as e:
        print(f"[llm_api] OpenRouter HTTP error: {e} / {getattr(e, 'response', None)}")
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "backend": "openrouter_err", "model": API_MODEL_ID}
    except Exception as e:
        print(f"[llm_api] OpenRouter call failed: {e}")
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "backend": "openrouter_err", "model": API_MODEL_ID}

# ===================== Public Router (unchanged signature) =====================

def call_llm(prompt: str) -> Tuple[str, Dict]:
    """
    Public entrypoint (unchanged). Routes per config:
      - backend=local:       use local Qwen
      - backend=openrouter:  use OpenRouter API
      - backend=auto:        prefer local; if unavailable or CPU-only (and not allowed), fall back to API
    Returns (json_text, usage_dict).
    """
    backend = BACKEND
    if backend == "local":
        return _call_llm_local(prompt)

    if backend == "openrouter":
        return _call_llm_openrouter(prompt)

    # auto
    txt, usage = _call_llm_local(prompt)
    if txt:
        return txt, usage

    print("[llm_api] Falling back to OpenRouter (auto mode).")
    return _call_llm_openrouter(prompt)
