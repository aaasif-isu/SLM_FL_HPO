# code/agent/policy_adapter.py

import os
import torch
from dataclasses import dataclass
from typing import Any, Optional, Dict, List

tokenizer = None
_base = None                 # frozen base model
_model = None                # PEFT-wrapped model hosting multiple adapters
_peft_ok = False

# Per-adapter state (keyed by normalized adapter key)
_optimizers: Dict[str, torch.optim.Optimizer] = {}
_round_counter: Dict[str, int] = {}
_active_key: Optional[str] = None
_adapters: Dict[str, bool] = {}   # existence map

@dataclass
class Cfg:
    enabled: bool = True
    adapter_mode: str = "per_cluster"  # "per_cluster" | "single"
    lora_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    step_lr: float = 5e-5
    max_grad_norm: float = 1.0
    kl_max: float = 0.05
    every_k_rounds: int = 1
    target_modules: Optional[List[str]] = None

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

# ---------- key normalization & matching helpers ----------

def _norm_once(key: Any) -> str:
    """
    Canonicalize to a single string, but DO NOT double-prefix.
    Accepts raw ints/strings; returns:
      - "global" as-is
      - "cluster:<int>" for ints and numeric strings
      - if already looks like "cluster:<...>", return as-is
    """
    if isinstance(key, str):
        if key == "global" or key.startswith("cluster:"):
            return key
        try:
            return f"cluster:{int(key)}"
        except Exception:
            return f"cluster:{key}"
    # non-str (e.g., int, numpy scalar)
    try:
        return f"cluster:{int(key)}"
    except Exception:
        return f"cluster:{str(key)}"

def _is_param_of_adapter(param_name: str, k: str) -> bool:
    """
    True iff this parameter belongs to LoRA adapter 'k'.
    Handles common PEFT patterns:
      lora_A.<k>.weight / lora_B.<k>.weight
      and legacy patterns with '... .<k>.' segments.
    """
    if "lora_" not in param_name:
        return False
    return (
        f".{k}." in param_name
        or param_name.endswith(f".{k}.weight")
        or param_name.endswith(f".{k}.bias")
        or f"lora_A.{k}" in param_name
        or f"lora_B.{k}" in param_name
    )

# ---------- public init / access ----------

def init_adapter_runtime(base_model, tok, cfg: dict):
    """
    Initialize the adaptation runtime.

    - If cfg.enabled == False (aka use_lora: false), run the frozen base model.
    - If enabled == True, wrap the base with PEFT LoRA once.
      Adapters are created lazily per key in set_active_adapter_key().
    - adapter_mode ("per_cluster" | "single") is stored in _cfg and used by set_active_adapter_key().
    """
    global tokenizer, _base, _model, _cfg, _peft_ok, _optimizers, _round_counter, _adapters, _active_key
    tokenizer = tok
    _base = base_model.eval() if base_model is not None else None
    _cfg = Cfg(**{**_cfg.__dict__, **(cfg or {})})

    # reset state on re-init
    _optimizers.clear()
    _round_counter.clear()
    _adapters.clear()
    _active_key = None

    if _base is None:
        _model = None
        _peft_ok = False
        print("[Adapter] No base model -> adapters unavailable.")
        return
    
    # If adapters are disabled, just run the frozen base
    if not _cfg.enabled:
        _model = _base
        _peft_ok = False
        print("[Adapter] Adapters disabled -> running frozen base.")
        return

    try:
        from peft import get_peft_model, LoraConfig, TaskType
        targets = _cfg.target_modules

        if targets is None:
            print("[Adapter] WARNING: target_modules is None; using PEFT defaults.")
        else:
            print(f"[Adapter] Using target_modules={targets}")


        lcfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=_cfg.lora_r, lora_alpha=_cfg.lora_alpha, lora_dropout=_cfg.lora_dropout,
            target_modules=targets,bias="none",
        )
        _model = get_peft_model(_base, lcfg)  # creates an initial "default" adapter

        # Freeze EVERYTHING by default (including lora_*). Re-enable only the active adapter later.
        for _, p in _model.named_parameters():
            p.requires_grad = False

        _model.eval()
        _peft_ok = True
        print(f"[Adapter] PEFT ready (adapter capable). mode={_cfg.adapter_mode}")

    except Exception as e:
        _model = _base
        _peft_ok = False
        print(f"[Adapter] PEFT unavailable -> running frozen. ({e})")

def get_infer_model():
    return _model

# ---------- adapter creation / switching ----------

def _ensure_adapter_for_key(raw_key: Any):
    """
    Create a LoRA adapter and its optimizer for a new key if missing.
    Accepts a raw key. Normalizes it exactly once.
    """
    global _adapters, _optimizers, _round_counter

    if not _peft_ok:
        return

    k = _norm_once(raw_key)
    if k in _adapters:
        return

    from peft import LoraConfig, TaskType

    targets = _cfg.target_modules
    if targets is None:
        print(f"[Adapter] WARNING: target_modules is None when creating adapter {k}; using PEFT defaults.")
    else:
        print(f"[Adapter] Creating adapter {k} with target_modules={targets}")

    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=_cfg.lora_r,
        lora_alpha=_cfg.lora_alpha,
        lora_dropout=_cfg.lora_dropout,
        target_modules=targets,
        bias="none",
    )

    try:
        if hasattr(_model, "add_adapter"):
            _model.add_adapter(adapter_name=k, peft_config=lcfg)
        else:
            print(f"[Adapter] WARNING: add_adapter not available; using single adapter for {k}")
    except Exception as e:
        print(f"[Adapter] add_adapter failed for key={k}: {e}")

    _set_trainable_for_adapter(k)

    # Build optimizer over CURRENT trainable params (this adapter only) in fp32
    params = [p for _, p in _model.named_parameters() if p.requires_grad]
    for p in params:
        if p.data.dtype in (torch.float16, torch.bfloat16):
            p.data = p.data.float()
    _optimizers[k] = torch.optim.AdamW(params, lr=_cfg.step_lr, eps=1e-6)
    _round_counter[k] = 0
    _adapters[k] = True
    print(f"[Adapter] Created LoRA adapter for key={k}")



def _ensure_adapter_for_key_old(raw_key: Any):
    """
    Create a LoRA adapter and its optimizer for a new key if missing.
    Accepts a raw key. Normalizes it exactly once.
    """
    global _adapters, _optimizers, _round_counter

    if not _peft_ok:
        return

    k = _norm_once(raw_key)
    if k in _adapters:
        return

    from peft import LoraConfig, TaskType
    target = _cfg.target_modules
    if targets is None:
        print(f"[Adapter] WARNING: target_modules is None when creating adapter {k}; using PEFT defaults.")
    else:
        print(f"[Adapter] Creating adapter {k} with target_modules={targets}")


    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=_cfg.lora_r, lora_alpha=_cfg.lora_alpha, lora_dropout=_cfg.lora_dropout,
        target_modules=targets,bias="none",
    )
    try:
        if hasattr(_model, "add_adapter"):
            _model.add_adapter(adapter_name=k, peft_config=lcfg)
        else:
            print(f"[Adapter] WARNING: add_adapter not available; using single adapter for {k}")
    except Exception as e:
        print(f"[Adapter] add_adapter failed for key={k}: {e}")

    _set_trainable_for_adapter(k)

    # Build optimizer over CURRENT trainable params (this adapter only) in fp32
    params = [p for _, p in _model.named_parameters() if p.requires_grad]
    for p in params:
        if p.data.dtype in (torch.float16, torch.bfloat16):
            p.data = p.data.float()
    _optimizers[k] = torch.optim.AdamW(params, lr=_cfg.step_lr, eps=1e-6)
    _round_counter[k] = 0
    _adapters[k] = True
    print(f"[Adapter] Created LoRA adapter for key={k}")

def _set_trainable_for_adapter(key_any: Any):
    """
    Freeze all params, then enable requires_grad only for the named adapter's LoRA params.
    Accepts either raw or normalized; normalizes exactly once.
    """
    k = _norm_once(key_any)

    # Switch active adapter for forward calls
    if hasattr(_model, "set_adapter"):
        try:
            _model.set_adapter(k)
        except Exception as e:
            # Helpful diagnostic
            print(f"[Adapter] WARNING: set_adapter({k}) failed: {e}")

    # Freeze all; then unfreeze only this adapter’s LoRA params
    for n, p in _model.named_parameters():
        if "lora_" in n:
            p.requires_grad = _is_param_of_adapter(n, k)
        else:
            p.requires_grad = False

    num_t = sum(
        p.numel()
        for _, p in _model.named_parameters()
        if p.requires_grad
    )
    print(f"[Adapter] Active adapter={k} trainable params={num_t}")

    

    # Optional: sanity print
    # trainables = [n for n, p in _model.named_parameters() if p.requires_grad]
    # print(f"[ADAPTER-SWITCH] key={k} num_trainables={len(trainables)} sample={trainables[:3]}")

def set_active_adapter_key(raw_key: Any):
    """
    Public API: activate (and if needed, create) the adapter for this key.
    Honors config:
      - _cfg.enabled (aka use_lora)
      - _cfg.adapter_mode: "per_cluster" | "single"
    """
    global _active_key

    # If LoRA is disabled or PEFT isn't available, run frozen base (no-op)
    if not getattr(_cfg, "enabled", True) or not _peft_ok or _model is None:
        _active_key = None
        return

    # Decide the *raw* key we want, then normalize exactly once here
    raw_actual = "global" if getattr(_cfg, "adapter_mode", "per_cluster") == "single" else raw_key
    k = _norm_once(raw_actual)

    # Fast path: already active
    if _active_key == k:
        return

    # Ensure the adapter exists and is the only trainable one
    _ensure_adapter_for_key(k)          # k is already normalized
    _set_trainable_for_adapter(k)       # use same normalized k
    _active_key = k

# ---------- tiny on-the-fly policy update ----------

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

def policy_update(
    *,
    prompt: str,
    response: str,
    reward: float,
    lyapunov_pass: bool,
    require_lyapunov: bool = False,  # only block if caller opts in
):
    """
    Tiny LoRA step for the CURRENT active adapter (per-cluster), scaled by reward,
    with an optional Lyapunov gate and a KL trust-region.
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

    # Optional Lyapunov gate (only if the caller requires it)
    if require_lyapunov and not lyapunov_pass:
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
        (p for _, p in _model.named_parameters() if p.requires_grad),
        _cfg.max_grad_norm
    )
    optimizer.step()
    _model.eval()

    # KL proxy (magnitude only)
    with torch.no_grad():
        new = _model(**inputs, labels=labels)
        new_logp = float(-new.loss.item() * max(n_tok, 1))

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

# --- HPO call-gating (independent of adapter update frequency) ---

_hpo_last_update_round: Dict[Any, int] = {}

def should_update_hps_for_client(
    *,
    client_id: int,
    round_idx: int,
    delta_acc: float,
    lyapunov_pass: bool,
    min_round_gap: int = 1,
    min_delta: float = 0.0,
    require_lyapunov: bool = False,
) -> bool:
    last = _hpo_last_update_round.get(client_id, -10**9)

    # 1) frequency gate
    if (round_idx - last) < int(min_round_gap):
        print(f"[HPO-GATE] freq-block: cid={client_id} round={round_idx} last={last} gap={min_round_gap}")
        return False

    # 2) improvement gate
    if abs(float(delta_acc)) < float(min_delta):
        print(f"[HPO-GATE] delta-block: cid={client_id} |Δacc|={abs(delta_acc):.4f} < {min_delta}")
        return False

    # 3) stability gate
    if require_lyapunov and not bool(lyapunov_pass):
        print(f"[HPO-GATE] lyapunov-block: cid={client_id} pass={lyapunov_pass}")
        return False

    return True

def mark_hpo_updated(client_id: int, round_idx: int) -> None:
    _hpo_last_update_round[client_id] = int(round_idx)



# --- Saving PEFT / LoRA state for later evaluation ---
from pathlib import Path

def save_all_lora_adapters(save_dir: str) -> None:
    """
    Save the current PEFT LoRA model (with all adapters) + tokenizer
    so we can reload and evaluate later.

    This assumes:
      - _peft_ok is True
      - _model is a PEFT-wrapped causal LM
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if not _peft_ok or _model is None:
        print("[Adapter-SAVE] No PEFT model with adapters; nothing to save.")
        return

    print(f"[Adapter-SAVE] Saving LoRA PEFT model (all adapters) to: {save_dir}")
    try:
        _model.save_pretrained(save_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)
    except Exception as e:
        print(f"[Adapter-SAVE] ERROR while saving LoRA adapters: {e}")
