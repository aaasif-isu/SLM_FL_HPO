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
from .policy_adapter import (
    init_adapter_runtime,
    get_infer_model,
    set_active_adapter_key,
    policy_update,
)

# ===================== Config & ENV =====================
load_dotenv()

CFG = shared_state.CONFIG or {}

BACKEND = (CFG.get("agents", {}).get("backend") or "local").lower().strip()
SLM_MODEL_ID = CFG.get("agents", {}).get("slm_model", "Qwen/Qwen2.5-0.5B-Instruct")
API_MODEL_ID = CFG.get("agents", {}).get("api_model", "openai/gpt-4o-mini")
LOCAL_ALLOW_CPU = bool(CFG.get("agents", {}).get("local_allow_cpu", False))
EXPLICIT_CUDA_DEVICE = CFG.get("agents", {}).get("cuda_device", None)


def refresh_from_shared_state() -> None:
    """
    Sync llm_api's local CFG and agent settings from shared_state.CONFIG.
    Call this *after* main() sets shared_state.CONFIG.
    """
    global CFG, BACKEND, SLM_MODEL_ID, API_MODEL_ID, LOCAL_ALLOW_CPU, EXPLICIT_CUDA_DEVICE

    CFG = shared_state.CONFIG or {}
    agents_cfg = CFG.get("agents", {})

    BACKEND = (agents_cfg.get("backend") or "local").lower().strip()
    SLM_MODEL_ID = agents_cfg.get("slm_model", "Qwen/Qwen2.5-0.5B-Instruct")
    API_MODEL_ID = agents_cfg.get("api_model", "openai/gpt-4o-mini")
    LOCAL_ALLOW_CPU = bool(agents_cfg.get("local_allow_cpu", False))
    EXPLICIT_CUDA_DEVICE = agents_cfg.get("cuda_device", None)

    print(f"[llm_api] Config refreshed. use_lora={CFG.get('model', {}).get('use_lora', None)} backend={BACKEND}")


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_SITE    = os.getenv("OPENROUTER_SITE", "http://localhost:3000")
OPENROUTER_APP     = os.getenv("OPENROUTER_APP", "FedHPO")

SYSTEM_JSON_ONLY = (
    "You are a strict JSON generator. Return ONLY a single JSON object. "
    "The ONLY keys allowed are 'reasoning', 'hps'. The 'hps' key MUST "
    "contain 'client', 'server', and 'mu' dictionaries/values."
)

ANTI_STALE_HINT = (
    "Policy:\n"
    "- Avoid repeating identical hyperparameters across rounds.\n"
    "- Adjust learning_rate, weight_decay, local_epochs, or batch_size logically.\n"
    "- Always output a single valid JSON object."
)

# ===================== Helper functions =====================
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        first_newline = s.find("\n")
        s = s[first_newline + 1 :] if first_newline != -1 else s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

def _first_json_object(s: str) -> Optional[str]:
    start = -1; depth = 0; in_str = False; esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        else:
            if ch == '"': in_str = True; continue
            if ch == "{":
                if depth == 0: start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        return s[start : i + 1]
    return None

def _sanitize_to_json(text: str) -> str:
    s = text.strip().split("<|im_end|>", 1)[0].strip()
    s = _strip_code_fences(s)
    obj = _first_json_object(s)
    return obj.strip() if obj else s

def _safe_set_pad(tok: AutoTokenizer):
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

# ===================== Device logic =====================
def _choose_slm_device() -> Optional[torch.device]:
    """
    Policy:
      - If >= 2 GPUs:
          * FL training assumed on cuda:0
          * SLM (Qwen + LoRA) goes to cuda:1
      - If exactly 1 GPU:
          * FL training on cuda:0
          * SLM on CPU if LOCAL_ALLOW_CPU else disabled (None)
      - If 0 GPUs:
          * SLM on CPU if LOCAL_ALLOW_CPU else disabled (None)

    EXPLICIT_CUDA_DEVICE (agents.cuda_device) is treated as a manual override.
    """
    try:
        num_gpus = torch.cuda.device_count()

        # 0) No CUDA at all
        if num_gpus == 0:
            if LOCAL_ALLOW_CPU:
                print("[llm_api] No CUDA available; falling back to CPU for SLM.")
                return torch.device("cpu")
            else:
                print("[llm_api] No CUDA available and CPU fallback disabled.")
                return None

        # 1) Manual override from config
        if EXPLICIT_CUDA_DEVICE is not None:
            idx = int(EXPLICIT_CUDA_DEVICE)
            if 0 <= idx < num_gpus:
                print(f"[llm_api] Using explicit cuda:{idx} for SLM.")
                return torch.device(f"cuda:{idx}")
            else:
                print(f"[llm_api] cuda_device={idx} invalid (device_count={num_gpus}); ignoring override.")

        # 2) Automatic policy
        if num_gpus >= 2:
            print("[llm_api] Using cuda:1 for SLM (cuda:0 reserved for FL training).")
            return torch.device("cuda:1")
        else:  # num_gpus == 1
            if LOCAL_ALLOW_CPU:
                print("[llm_api] Only one GPU visible; using CPU for SLM to keep cuda:0 for FL.")
                return torch.device("cpu")
            else:
                print("[llm_api] Only one GPU visible and CPU fallback disabled; skipping local SLM.")
                return None

    except Exception as e:
        print(f"[llm_api] Device selection error: {e}")
        return None


def _choose_slm_device_old() -> Optional[torch.device]:
    try:
        num_gpus = torch.cuda.device_count()
        if EXPLICIT_CUDA_DEVICE is not None:
            idx = int(EXPLICIT_CUDA_DEVICE)
            if num_gpus > idx:
                return torch.device(f"cuda:{idx}")
            print(f"[llm_api] cuda_device={idx} invalid; fallback.")
        if num_gpus >= 2:
            return torch.device("cuda:1")
        elif num_gpus == 1:
            return torch.device("cpu") if LOCAL_ALLOW_CPU else None
        else:
            return torch.device("cpu") if LOCAL_ALLOW_CPU else None
    except Exception as e:
        print(f"[llm_api] Device selection error: {e}")
        return None

# ===================== Local model =====================
_tokenizer: Optional[AutoTokenizer] = None
_base_model = None
_local_ready = False

def _maybe_init_local() -> None:
    global _tokenizer, _base_model, _local_ready
    if _local_ready:
        return

    target_device = _choose_slm_device()
    if target_device is None:
        print("[llm_api] Skipping local SLM init.")
        return

    try:
        print(f"[llm_api] Loading local LLM: {SLM_MODEL_ID} → {target_device}")
        tok = AutoTokenizer.from_pretrained(SLM_MODEL_ID, trust_remote_code=True)
        _safe_set_pad(tok)

        model = AutoModelForCausalLM.from_pretrained(
            SLM_MODEL_ID, trust_remote_code=True
        ).to(target_device).eval()

        m = CFG.get("model", {})
        l = m.get("lora", {}) or {}
        adapter_cfg = {
            "enabled": bool(m.get("use_lora", False)),
            "adapter_mode": m.get("adapter_mode", "per_cluster"),
            "lora_r": int(l.get("r", 4)),
            "lora_alpha": int(l.get("alpha", 4)),
            "lora_dropout": float(l.get("dropout", 0.05)),
            "step_lr": float(l.get("step_lr", 2e-5)),
            "max_grad_norm": float(l.get("max_grad_norm", 0.5)),
            "kl_max": float(l.get("kl_max", 0.01)),
            "every_k_rounds": int(l.get("every_k_rounds", 3)),

            "target_modules": l.get("target_modules", None),

        }

        init_adapter_runtime(base_model=model, tok=tok, cfg=adapter_cfg)
        _tokenizer, _base_model, _local_ready = tok, model, True

        print(f"[llm_api] LoRA enabled={adapter_cfg['enabled']} mode={adapter_cfg['adapter_mode']}")
        print(f"[llm_api] LoRA adapters will update on: {target_device}")

    except Exception as e:
        print(f"[llm_api] Local init failed: {e}")

# ===================== Local inference =====================
def _call_llm_local(prompt: str) -> Tuple[str, Dict]:
    _maybe_init_local()
    model = get_infer_model()
    tok = _tokenizer
    if model is None or tok is None:
        return "", {"backend": "local_unavailable"}

    try:
        start = time.time()
        messages = [
            {"role": "system", "content": SYSTEM_JSON_ONLY},
            {"role": "user", "content": ANTI_STALE_HINT + "\n\n" + prompt},
        ]
        chat_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        device = next(model.parameters()).device
        inputs = tok(chat_text, return_tensors="pt", padding=True, truncation=True).to(device)

        # --- new sampling logic ---
        use_lora = bool((CFG.get("model") or {}).get("use_lora", False))
        gen_common = dict(max_new_tokens=512, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)

        if use_lora:
            torch.manual_seed(2025)  # reproducible stochasticity
            gen_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.3,
                top_p=0.95,
                **gen_common,
            )
        else:
            gen_ids = model.generate(
                **inputs,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                **gen_common,
            )

        new_tokens = gen_ids[0, inputs.input_ids.shape[1]:]
        raw_text = tok.decode(new_tokens, skip_special_tokens=True)
        cleaned = _sanitize_to_json(raw_text)
        end = time.time()
        usage = {
            "prompt_tokens": int(inputs.input_ids.numel()),
            "completion_tokens": int(new_tokens.numel()),
            "latency_ms": (end - start) * 1000.0,
            "backend": "local",
            "model": SLM_MODEL_ID,
            "device": str(device),
        }
        return cleaned, usage
    except Exception as e:
        print(f"[llm_api] Local inference error: {e}")
        return "", {"backend": "local_err", "model": SLM_MODEL_ID}

# ===================== OpenRouter API =====================
def _call_llm_openrouter(prompt: str) -> Tuple[str, Dict]:
    if not OPENROUTER_API_KEY:
        print("[llm_api] OPENROUTER_API_KEY not set.")
        return "", {"backend": "openrouter_unavailable"}
    try:
        start = time.time()
        messages = [
            {"role": "system", "content": SYSTEM_JSON_ONLY},
            {"role": "user", "content": ANTI_STALE_HINT + "\n\n" + prompt},
        ]
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": OPENROUTER_SITE,
            "X-Title": OPENROUTER_APP,
            "Content-Type": "application/json",
        }
        data = {"model": API_MODEL_ID, "messages": messages, "temperature": 0.0, "top_p": 1.0, "max_tokens": 512}
        resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, data=json.dumps(data), timeout=60)
        resp.raise_for_status()
        j = resp.json()
        content = j.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        cleaned = _sanitize_to_json(content)
        usage_api = j.get("usage", {}) or {}
        end = time.time()
        usage = {
            "prompt_tokens": int(usage_api.get("prompt_tokens", 0)),
            "completion_tokens": int(usage_api.get("completion_tokens", 0)),
            "latency_ms": (end - start) * 1000.0,
            "backend": "openrouter",
            "model": API_MODEL_ID,
        }
        return cleaned, usage
    except Exception as e:
        print(f"[llm_api] OpenRouter error: {e}")
        return "", {"backend": "openrouter_err", "model": API_MODEL_ID}

# ===================== Public router =====================
def call_llm(prompt: str) -> Tuple[str, Dict]:
    backend = BACKEND
    if backend == "local":
        return _call_llm_local(prompt)
    if backend == "openrouter":
        return _call_llm_openrouter(prompt)
    txt, usage = _call_llm_local(prompt)
    if txt:
        return txt, usage
    print("[llm_api] Falling back to OpenRouter.")
    return _call_llm_openrouter(prompt)


# --- Saving baseline SLM (no LoRA) ---


from pathlib import Path

def save_slm_baseline(save_dir: str) -> None:
    """
    Save the baseline SLM (no LoRA) + tokenizer for later evaluation.
    Uses the _base_model that we keep around in llm_api.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    global _base_model, _tokenizer
    if _base_model is None:
        print("[LLM-SAVE] No base SLM loaded; nothing to save.")
        return

    model_to_save = _base_model
    print(f"[LLM-SAVE] Saving baseline SLM (no LoRA) to: {save_dir}")
    try:
        model_to_save.save_pretrained(save_dir)
        if _tokenizer is not None:
            _tokenizer.save_pretrained(save_dir)
    except Exception as e:
        print(f"[LLM-SAVE] ERROR while saving baseline SLM: {e}")
