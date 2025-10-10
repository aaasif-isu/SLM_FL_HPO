# code/agent/policy_adapter.py
import torch
from dataclasses import dataclass
from typing import Any, Optional, Dict

tokenizer = None
_base = None                 # frozen base model
_model = None                # PEFT-wrapped model hosting multiple adapters
_peft_ok = False

# Per-adapter (keyed by cluster_id) state
_optimizers: Dict[Any, torch.optim.Optimizer] = {}
_round_counter: Dict[Any, int] = {}
_active_key: Any = None
_adapters: Dict[Any, bool] = {}   # existence map

@dataclass
class Cfg:
    enabled: bool = True
    lora_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    step_lr: float = 5e-5
    max_grad_norm: float = 1.0
    kl_max: float = 0.05
    every_k_rounds: int = 1

_cfg = Cfg()

# --- local copies to avoid circular import with llm_api ---
SYSTEM_JSON_ONLY_LOCAL = (
    "You are a function that returns JSON only. "
    "Output must be a single valid JSON object. "
    "Do not include markdown, backticks, or any text outside JSON."
)
ANTI_STALE_HINT_LOCAL = (
    "Policy:\n"
    "- If this is NOT the first round and prior hyperparameters appear in the context/history, "
    "avoid repeating identical values; make at least one meaningful change within allowed constraints.\n"
    "- If train accuracy is much higher than test accuracy, reduce learning_rate (e.g., x0.5) OR increase weight_decay (e.g., x2), "
    "OR decrease local_epochs by 1 (not below min), OR lower batch_size to the next allowed option; pick exactly one change.\n"
    "- If both train and test accuracies are low, increase learning_rate (e.g., x1.5) within the allowed max.\n"
    "- Never copy example values verbatim. Always personalize based on history/peer signals.\n"
    "- Output must remain a single valid JSON object."
)

def init_adapter_runtime(base_model, tok, cfg: dict):
    """
    Wrap the frozen base with PEFT once.
    New adapters are created lazily per cluster via set_active_adapter_key(key).
    """
    global tokenizer, _base, _model, _cfg, _peft_ok
    tokenizer = tok
    _base = base_model.eval()
    _cfg = Cfg(**{**_cfg.__dict__, **cfg})

    try:
        from peft import get_peft_model, LoraConfig, TaskType
        # Create an initial PEFT wrapper with a tiny default adapter we won't train.
        lcfg = LoraConfig(task_type=TaskType.CAUSAL_LM,
                          r=_cfg.lora_r, lora_alpha=_cfg.lora_alpha, lora_dropout=_cfg.lora_dropout)
        _model = get_peft_model(_base, lcfg)             # creates an initial adapter ("default")
        # Freeze everything by default (we’ll enable only the active adapter’s params)
        for n, p in _model.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False
        _model.eval()
        _peft_ok = True
        print("[Adapter] PEFT ready (multi-adapter capable).")
    except Exception as e:
        _model = _base
        _peft_ok = False
        print(f"[Adapter] PEFT unavailable -> running frozen. ({e})")

def get_infer_model():
    return _model

def _ensure_adapter_for_key(key: Any):
    """
    Create a LoRA adapter and its optimizer for a new key (cluster_id) if missing.
    """
    global _adapters, _optimizers, _round_counter

    if not _peft_ok:
        # No PEFT — nothing to do
        return

    if key in _adapters:
        return

    from peft import LoraConfig, TaskType

    # Add a named adapter to the existing PEFT model
    lcfg = LoraConfig(task_type=TaskType.CAUSAL_LM,
                      r=_cfg.lora_r, lora_alpha=_cfg.lora_alpha, lora_dropout=_cfg.lora_dropout)
    try:
        # PeftModel exposes add_adapter in recent versions
        if hasattr(_model, "add_adapter"):
            _model.add_adapter(adapter_name=str(key), peft_config=lcfg)
        else:
            # Fallback: if add_adapter is not present, we still have at least the default adapter
            # (In very old PEFT, multi-adapter might not be supported.)
            print(f"[Adapter] WARNING: add_adapter not available; reusing single adapter for key={key}")
            _adapters[key] = True
            if key not in _optimizers:
                # Build optimizer over current trainable params (likely the default adapter)
                params = [p for n, p in _model.named_parameters() if p.requires_grad]
                _optimizers[key] = torch.optim.AdamW(params, lr=_cfg.step_lr)
                _round_counter[key] = 0
            return
    except Exception as e:
        print(f"[Adapter] add_adapter failed for key={key}: {e}")
        # Fallback as above
        _adapters[key] = True
        if key not in _optimizers:
            params = [p for n, p in _model.named_parameters() if p.requires_grad]
            _optimizers[key] = torch.optim.AdamW(params, lr=_cfg.step_lr)
            _round_counter[key] = 0
        return

    # Mark only this adapter as trainable
    _set_trainable_for_adapter(key)

    # Create optimizer for this adapter’s params
    params = [p for n, p in _model.named_parameters() if p.requires_grad]
    _optimizers[key] = torch.optim.AdamW(params, lr=_cfg.step_lr)
    _round_counter[key] = 0
    _adapters[key] = True
    print(f"[Adapter] Created LoRA adapter for key={key}")

def _set_trainable_for_adapter(key: Any):
    """
    Freeze all LoRA params, then enable requires_grad only for the named adapter.
    """
    # Switch active adapter for forward calls
    if hasattr(_model, "set_adapter"):
        _model.set_adapter(str(key))

    # Freeze all params, then unfreeze only this adapter’s LoRA params
    for n, p in _model.named_parameters():
        if "lora_" in n:
            p.requires_grad = (f".{key}." in n) or (n.endswith(f".{key}.weight")) or (n.endswith(f".{key}.bias"))
        else:
            p.requires_grad = False

def set_active_adapter_key(key: Any):
    """
    Public API: activate (and if needed, create) the adapter for this cluster key.
    """
    global _active_key
    if not _peft_ok:
        _active_key = None
        return
    _ensure_adapter_for_key(key)
    _set_trainable_for_adapter(key)
    _active_key = key

def _build_chat_io(prompt: str, response: Optional[str] = None):
    """
    Build the SAME chat template as call_llm, then (optionally) append response for teacher forcing.
    """
    assert tokenizer is not None and get_infer_model() is not None
    wrapped_prompt = ANTI_STALE_HINT_LOCAL + "\n\n" + prompt
    messages = [
        {"role": "system", "content": SYSTEM_JSON_ONLY_LOCAL + " Prefer adjustments over repetition. Never copy example values."},
        {"role": "user", "content": wrapped_prompt},
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(get_infer_model().parameters()).device
    chat = tokenizer(chat_text, return_tensors="pt").to(device)

    if response is None:
        return chat, None, 0

    resp = tokenizer(response, return_tensors="pt").to(device)
    input_ids = torch.cat([chat["input_ids"], resp["input_ids"]], dim=1)
    attn_mask = torch.cat([chat["attention_mask"], resp["attention_mask"]], dim=1)

    labels = input_ids.clone()
    n_prompt = chat["input_ids"].shape[1]
    labels[:, :n_prompt] = -100

    return {"input_ids": input_ids, "attention_mask": attn_mask}, labels, int(resp["input_ids"].shape[1])

def policy_update(*, prompt: str, response: str, reward: float, lyapunov_pass: bool):
    """
    Tiny LoRA step for the CURRENT active adapter (per-cluster), scaled by reward,
    with a Lyapunov gate and a KL trust-region.
    """
    info = {"updated": False, "reason": "", "kl": 0.0}

    # If adapters are off/frozen
    if not _cfg.enabled or tokenizer is None or _model is None:
        info["reason"] = "adapter_unavailable"
        return info
    if not _peft_ok:
        info["reason"] = "peft_unavailable"
        return info
    if _active_key is None or _active_key not in _optimizers:
        info["reason"] = "no_active_adapter"
        return info

    # Frequency gate (per-adapter counter)
    _round_counter[_active_key] += 1
    if (_round_counter[_active_key] % _cfg.every_k_rounds) != 0:
        info["reason"] = "frequency_gate"
        return info

    if not lyapunov_pass:
        info["reason"] = "lyapunov_block"
        return info

    # Use current active adapter's optimizer
    optimizer = _optimizers[_active_key]

    inputs, labels, n_tok = _build_chat_io(prompt, response)
    if n_tok <= 0:
        info["reason"] = "empty_response_tokens"
        return info

    # Snapshot pre-update LoRA params **for this adapter only** (requires_grad=True)
    pre = {n: p.detach().clone() for n, p in _model.named_parameters() if p.requires_grad}

    # Old log-prob
    with torch.no_grad():
        old = _model(**inputs, labels=labels)
        old_logp = float(-old.loss.item() * max(n_tok, 1))

    # REINFORCE-style tiny step
    _model.train()
    optimizer.zero_grad(set_to_none=True)
    out = _model(**inputs, labels=labels)
    logp = -out.loss * n_tok
    scaled = float(max(min(reward, 1.0), -1.0))  # clamp for safety
    loss = - scaled * logp
    print(f"[AdapterDebug] key={_active_key} reward_raw={reward:.4f} reward_scaled={scaled:.4f}")

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        (p for n, p in _model.named_parameters() if p.requires_grad),
        _cfg.max_grad_norm
    )
    optimizer.step()
    _model.eval()

    # KL proxy (sign fixed: positive when moving away from old policy on this response)

    with torch.no_grad():
        new = _model(**inputs, labels=labels)
        new_logp = float(-new.loss.item() * max(n_tok, 1))

    # Non-negative trust-region proxy (magnitude only)
    kl_proxy = abs((old_logp - new_logp) / max(n_tok, 1))
    info["kl"] = float(kl_proxy)

    if kl_proxy > _cfg.kl_max:
        # Project back toward pre-update weights for THIS adapter
        tau = max(0.0, min(1.0, _cfg.kl_max / kl_proxy))
        with torch.no_grad():
            for n, p in _model.named_parameters():
                if p.requires_grad and n in pre:
                    p.data.copy_(tau * p.data + (1.0 - tau) * pre[n])
        info["reason"] = "trust_region_project"
        return info

    info.update({"updated": True, "reason": "ok"})
    return info

