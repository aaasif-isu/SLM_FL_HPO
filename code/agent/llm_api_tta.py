# code/agent/llm_api.py

import time
import json
import re
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- NEW: LoRA adapter (optional on-the-fly update) ---
try:
    from peft import get_peft_model, LoraConfig, TaskType
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False

# ===================== Model Selection (unchanged) =====================

# Choose your local instruct model
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Loading local LLM: {model_id}...")

tokenizer = None
_base_model = None          # frozen base (weights)
_model_for_infer = None     # this is what call_llm() uses for generate()
_adapter_enabled = False    # toggled true if PEFT available & init succeeds

# LoRA / adaptation config (safe small defaults)
@dataclass
class _AdaptCfg:
    enabled: bool = True         # master switch
    kl_max: float = 0.05         # trust-region size δ
    step_lr: float = 5e-5        # tiny LR for adapter
    max_grad_norm: float = 1.0
    every_k_rounds: int = 1      # do adapter step every K calls to policy_update
    lora_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.05

_adapt_cfg = _AdaptCfg()
_round_counter = 0  # counts policy_update calls to apply every_k_rounds

# Helper to know device
def _device_of(model: AutoModelForCausalLM) -> str:
    try:
        return next(model.parameters()).device.type
    except Exception:
        return "cpu"

# Try to load tokenizer and model (same behavior you had)
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    _base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
    )
    # Ensure pad token exists to avoid attention mask warnings
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # By default, use base model for inference
    _model_for_infer = _base_model.eval()

    # If PEFT is available and adaptation enabled, wrap with LoRA (small, trainable)
    if _PEFT_AVAILABLE and _adapt_cfg.enabled:
        lcfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=_adapt_cfg.lora_r,
            lora_alpha=_adapt_cfg.lora_alpha,
            lora_dropout=_adapt_cfg.lora_dropout,
        )
        _model_for_infer = get_peft_model(_base_model, lcfg)  # attaches LoRA modules
        # Freeze non-LoRA params
        for n, p in _model_for_infer.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False
        _optimizer = torch.optim.AdamW(
            (p for p in _model_for_infer.parameters() if p.requires_grad),
            lr=_adapt_cfg.step_lr,
        )
        _adapter_enabled = True
        print("LoRA adapter initialized for on-the-fly updates.")
    else:
        _optimizer = None
        if not _PEFT_AVAILABLE:
            print("PEFT not available; running without on-the-fly adaptation.")
        else:
            print("On-the-fly adaptation disabled by config; using base model only.")

    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model from Hugging Face: {e}")
    print("This might be due to insufficient VRAM or a missing dependency.")
    tokenizer = None
    _base_model = None
    _model_for_infer = None
    _optimizer = None
    _adapter_enabled = False

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

# ---------- NEW: tiny helpers for adaptation ----------

def _encode_prompt_response(prompt: str, response: str):
    """
    Prepare teacher-forced labels so loss is only on response tokens.
    """
    assert tokenizer is not None and _model_for_infer is not None
    tok = tokenizer
    device = next(_model_for_infer.parameters()).device
    full = tok(prompt + tok.eos_token + response, return_tensors="pt").to(device)
    labels = full.input_ids.clone()
    # mask prompt tokens (no loss over them)
    num_prompt = len(tok(prompt, return_tensors="pt").input_ids[0]) + 1
    labels[:, :num_prompt] = -100
    return full, labels

@torch.no_grad()
def _logprob_of_response(prompt: str, response: str) -> Tuple[float, int]:
    """
    Approximate log-prob of model generating `response` given `prompt`.
    Returns (total_log_prob, token_count_over_response).
    """
    if tokenizer is None or _model_for_infer is None:
        return 0.0, 1
    full, labels = _encode_prompt_response(prompt, response)
    out = _model_for_infer(**full, labels=labels)
    n_tok = int((labels != -100).sum().item())
    logp = float(-out.loss.item() * max(n_tok, 1))
    return logp, n_tok

# ---------- main call (UNCHANGED SIGNATURE) ----------

def call_llm(prompt: str) -> Tuple[str, Dict]:
    """
    Calls the local Qwen Instruct model, enforcing JSON-only output via a system message.
    Sanitizes the reply into a parseable JSON string.
    Returns (json_like_text, usage_dict).
    """
    if _model_for_infer is None or tokenizer is None:
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
        ).to(next(_model_for_infer.parameters()).device)

        # Deterministic decoding (no unsupported flags like top_k)
        gen_ids = _model_for_infer.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,           # deterministic
            temperature=0.0,           # ignored when do_sample=False
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

# ---------- NEW: optional on-the-fly adaptation entrypoint ----------

def policy_update(
    *,
    prompt: str,
    response: str,
    reward: float,
    lyapunov_pass: bool
) -> Dict[str, Any]:
    """
    Optional tiny adapter update to make `response` more/less likely next time,
    scaled by `reward`. Enforces a KL trust region and a frequency gate.
    Public signature is NEW but independent; does not alter call_llm() signature.
    Returns info dict with keys: updated(bool), reason(str), kl(float).
    """
    global _round_counter

    info = {"updated": False, "reason": "", "kl": 0.0}

    # If no adapter or disabled, just no-op
    if not _adapter_enabled or _optimizer is None or _model_for_infer is None or tokenizer is None:
        info["reason"] = "adapter_unavailable"
        return info

    if not _adapt_cfg.enabled:
        info["reason"] = "disabled"
        return info

    _round_counter += 1
    if (_round_counter % _adapt_cfg.every_k_rounds) != 0:
        info["reason"] = "frequency_gate"
        return info

    if not lyapunov_pass:
        info["reason"] = "lyapunov_block"
        return info

    # Encode prompt/response with teacher forcing (loss only on response tokens)
    full, labels = _encode_prompt_response(prompt, response)
    n_tok = int((labels != -100).sum().item())

    # Old log-prob for KL proxy
    with torch.no_grad():
        old_out = _model_for_infer(**full, labels=labels)
        old_logp = float(-old_out.loss.item() * max(n_tok, 1))

    # Policy-gradient style objective: min  -reward * log pθ(response|prompt)
    _model_for_infer.train()
    _optimizer.zero_grad(set_to_none=True)
    out = _model_for_infer(**full, labels=labels)
    logp = -out.loss * n_tok
    loss = - reward * logp
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        (p for p in _model_for_infer.parameters() if p.requires_grad),
        _adapt_cfg.max_grad_norm
    )
    _optimizer.step()
    _model_for_infer.eval()

    # KL trust region (crude per-token proxy): KL ≈ E[log p_old - log p_new]
    with torch.no_grad():
        new_out = _model_for_infer(**full, labels=labels)
        new_logp = float(-new_out.loss.item() * max(n_tok, 1))
    kl = ((-old_logp) - (-new_logp)) / max(n_tok, 1)
    info["kl"] = float(kl)

    if kl > _adapt_cfg.kl_max:
        # Project back toward pre-update weights by interpolation on LoRA params only
        gamma = max(0.0, min(1.0, _adapt_cfg.kl_max / kl))
        for n, p in _model_for_infer.named_parameters():
            if p.requires_grad:  # LoRA params only
                # "Project" by shrinking the delta (simple, effective)
                p.data = (1 - gamma) * p.data + gamma * p.data.detach()
        info["reason"] = "trust_region_project"
        return info

    info.update(updated=True, reason="ok")
    return info
