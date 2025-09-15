#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal, version-agnostic DPO finetuning for Qwen using your dpo_flat.jsonl.

Input JSONL record (one per line):
{
  "prompt": "<user text prompt>",
  "chosen": "<assistant JSON string (preferred)>",
  "rejected": "<assistant JSON string (less preferred)>",
  "metadata": {...}   // optional, ignored for training
}

Why this is robust:
- No TRL version assumptions (we don't import TRL's trainers).
- Manual DPO loss & train loop.
- Only the reference model path is no_grad; policy path requires grad.

Requirements:
  pip install torch transformers peft bitsandbytes datasets

Suggested env vars (optional):
  MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
  DATA_PATH=/path/to/dpo_flat.jsonl
  OUT_DIR=./qwen_dpo_lora
"""

import os
import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import LoraConfig, get_peft_model
from datasets import load_dataset

try:
    from transformers import BitsAndBytesConfig
    HAVE_BNB = True
except Exception:
    HAVE_BNB = False


# -------------------------
# Config (env or defaults)
# -------------------------
MODEL_ID  = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_PATH = os.environ.get("DATA_PATH", "/u/aalasif/SLM_FL_HPO/code/parse/dpo_flat.jsonl")
OUT_DIR   = os.environ.get("OUT_DIR",  "/u/aalasif/SLM_FL_HPO/code/finetune/qwen_dpo_lora")

SEED      = int(os.environ.get("SEED", "42"))
MAX_LEN   = int(os.environ.get("MAX_LEN", "1536"))
BATCH     = int(os.environ.get("BATCH", "2"))
EPOCHS    = float(os.environ.get("EPOCHS", "2"))
LR        = float(os.environ.get("LR", "5e-5"))
BETA      = float(os.environ.get("BETA", "0.1"))   # DPO beta
GA        = int(os.environ.get("GA", "8"))         # gradient accumulation
USE_4BIT  = bool(int(os.environ.get("USE_4BIT", "1")))
BF16      = bool(int(os.environ.get("BF16", "1"))) # use bf16 if GPU supports


# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_str(x) -> str:
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)


def require_three_fields(rec: Dict[str, Any]) -> bool:
    """Basic sanity: need prompt / chosen / rejected all present and non-empty."""
    return bool(rec.get("prompt")) and bool(rec.get("chosen")) and bool(rec.get("rejected"))


# -------------------------
# Dataset
# -------------------------
class DPOFlatDataset(Dataset):
    """
    Each item provides:
      - prompt_text (wrapped with chat template and add_generation_prompt=True)
      - chosen_text (JSON string)
      - rejected_text (JSON string)

    We tokenize later in the collate for flexibility.
    """
    def __init__(self, path: str):
        # Robustly load jsonl
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Cannot find data at {path}")

        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if require_three_fields(obj):
                    rows.append({
                        "prompt": ensure_str(obj["prompt"]),
                        "chosen": ensure_str(obj["chosen"]),
                        "rejected": ensure_str(obj["rejected"]),
                        "metadata": obj.get("metadata", None),
                    })

        if not rows:
            raise ValueError(f"No valid DPO rows in {path}")

        print(f"[INFO] Loaded {len(rows)} DPO pairs from {path}")
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


# -------------------------
# Tokenization / Collate
# -------------------------
@dataclass
class DPOBatch:
    # Policy inputs
    pol_chosen_input_ids: torch.Tensor
    pol_chosen_attn: torch.Tensor
    pol_chosen_prompt_lens: torch.Tensor

    pol_rejected_input_ids: torch.Tensor
    pol_rejected_attn: torch.Tensor
    pol_rejected_prompt_lens: torch.Tensor

    # For reference model we just reuse the same tokenization


def apply_qwen_prompt(tokenizer, prompt: str) -> str:
    """
    Wrap a plain prompt as chat for Qwen and append assistant header.
    """
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )


def tokenize_pair(
    tokenizer,
    prompt_text: str,
    completion_text: str,
    max_len: int
) -> Tuple[List[int], List[int], int]:
    """
    Build (input_ids, attention_mask, prompt_len) for prompt + completion.
    We compute prompt_len using the tokenized prompt alone.
    """
    prompt_tok = tokenizer(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_tok["input_ids"])

    full_text = prompt_text + completion_text + tokenizer.eos_token
    full_tok = tokenizer(
        full_text, add_special_tokens=False,
        max_length=max_len, truncation=True
    )
    return full_tok["input_ids"], full_tok["attention_mask"], prompt_len


def dpo_collate_fn(tokenizer, max_len):
    def _collate(samples: List[Dict[str, Any]]) -> DPOBatch:
        pol_chosen_ids, pol_chosen_attn, pl_chosen = [], [], []
        pol_rej_ids, pol_rej_attn, pl_rej = [], [], []

        for s in samples:
            prompt_text = apply_qwen_prompt(tokenizer, s["prompt"])

            # POLICY (same tokenization used for reference model)
            ci, ca, cpl = tokenize_pair(tokenizer, prompt_text, s["chosen"], max_len)
            ri, ra, rpl = tokenize_pair(tokenizer, prompt_text, s["rejected"], max_len)

            pol_chosen_ids.append(ci)
            pol_chosen_attn.append(ca)
            pl_chosen.append(cpl)

            pol_rej_ids.append(ri)
            pol_rej_attn.append(ra)
            pl_rej.append(rpl)

        # Pad to max length within batch
        def pad_batch(seqs, pad_id):
            m = max(len(x) for x in seqs)
            out = []
            att = []
            for x in seqs:
                padn = m - len(x)
                out.append(x + [pad_id]*padn)
                att.append([1]*len(x) + [0]*padn)
            return torch.tensor(out, dtype=torch.long), torch.tensor(att, dtype=torch.long)

        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        pci, pca = pad_batch(pol_chosen_ids, pad_id)
        pri, pra = pad_batch(pol_rej_ids, pad_id)

        pl_chosen = torch.tensor(pl_chosen, dtype=torch.long)
        pl_rej = torch.tensor(pl_rej, dtype=torch.long)

        return DPOBatch(
            pol_chosen_input_ids=pci,
            pol_chosen_attn=pca,
            pol_chosen_prompt_lens=pl_chosen,
            pol_rejected_input_ids=pri,
            pol_rejected_attn=pra,
            pol_rejected_prompt_lens=pl_rej,
        )
    return _collate


# -------------------------
# Log-prob computation
# -------------------------
def sequence_logprob(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    """
    Sum log-probs over completion tokens (exclude prompt) for each sequence.

    input_ids: [B, T]
    attention_mask: [B, T]
    prompt_lens: [B] number of tokens belonging to the prompt for each sample
    return: [B] sum log-prob of completion tokens
    """
    # NOTE: DO NOT use torch.no_grad() here (we need gradients for the policy).
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, T, V]

    # Shift for next-token prediction
    logits = logits[:, :-1, :]             # [B, T-1, V]
    labels = input_ids[:, 1:]              # [B, T-1]
    attn   = attention_mask[:, 1:]         # [B, T-1]

    # Compute log-softmax
    logprobs = torch.log_softmax(logits, dim=-1)  # [B, T-1, V]
    # Gather chosen token log-probs
    token_lp = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    B, TL = labels.shape
    # Build a mask to include only completion tokens
    # First label index corresponding to completion is prompt_len (since labels are shifted)
    # -> logits index is prompt_len-1
    rng = torch.arange(TL, device=labels.device).unsqueeze(0).expand(B, TL)
    comp_mask = (rng >= (prompt_lens.unsqueeze(1) - 1).clamp(min=0)).long()  # [B, T-1]

    # Also respect original attention (in case of padding)
    comp_mask = comp_mask * attn

    sum_lp = (token_lp * comp_mask).sum(dim=-1)  # [B]
    return sum_lp


@torch.no_grad()
def sequence_logprob_ref(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    """Reference log-prob — identical to above but no gradients."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    attn   = attention_mask[:, 1:]

    logprobs = torch.log_softmax(logits, dim=-1)
    token_lp = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    B, TL = labels.shape
    rng = torch.arange(TL, device=labels.device).unsqueeze(0).expand(B, TL)
    comp_mask = (rng >= (prompt_lens.unsqueeze(1) - 1).clamp(min=0)).long()
    comp_mask = comp_mask * attn

    sum_lp = (token_lp * comp_mask).sum(dim=-1)
    return sum_lp


def dpo_loss(
    beta: float,
    pi_c: torch.Tensor,
    pi_r: torch.Tensor,
    ref_c: torch.Tensor,
    ref_r: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    """
    Standard DPO loss:
      L = - E[ log σ( β * ( (pi_c - pi_r) - (ref_c - ref_r) ) ) ]
    Also return a simple accuracy proxy: % where policy prefers chosen over rejected.
    """
    import torch.nn.functional as F
    advantages = (pi_c - pi_r) - (ref_c - ref_r)      # [B]
    logits = beta * advantages
    loss = -F.logsigmoid(logits).mean()
    acc = (advantages > 0).float().mean()
    return loss, acc


# -------------------------
# Main
# -------------------------
def main():
    set_seed(SEED)

    # 1) Data
    dataset = DPOFlatDataset(DATA_PATH)

    # 2) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) Models: policy (trainable via LoRA) + frozen reference
    bnb_cfg = None
    if USE_4BIT and HAVE_BNB:
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    policy = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_cfg,
    )

    # Enable LoRA on attention/MLP blocks
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(policy, lora_cfg)
    policy.train()

    # Reference model (no LoRA), frozen
    reference = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_cfg,
    )
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    # 4) Dataloader
    collate = dpo_collate_fn(tokenizer, MAX_LEN)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, collate_fn=collate)

    # 5) Optimizer
    optim = torch.optim.AdamW(policy.parameters(), lr=LR)

    # 6) Train
    device = next(policy.parameters()).device
    total_steps = math.ceil(len(loader) * EPOCHS / GA)
    print(f"[INFO] Kept {len(dataset)} pairs after validation & tokenization.")
    print(f"[INFO] Training for {EPOCHS} epochs, ~{total_steps} optimizer steps (GA={GA}).")

    step = 0
    running = 0.0
    running_acc = 0.0
    for epoch in range(int(math.ceil(EPOCHS))):
        for it, batch in enumerate(loader):

            # Move batch to device
            pci = batch.pol_chosen_input_ids.to(device)
            pca = batch.pol_chosen_attn.to(device)
            cpl = batch.pol_chosen_prompt_lens.to(device)

            pri = batch.pol_rejected_input_ids.to(device)
            pra = batch.pol_rejected_attn.to(device)
            rpl = batch.pol_rejected_prompt_lens.to(device)

            # Policy log-probs (requires grad)
            pi_c = sequence_logprob(policy, pci, pca, cpl)   # [B]
            pi_r = sequence_logprob(policy, pri, pra, rpl)   # [B]

            # Reference log-probs (no grad)
            ref_c = sequence_logprob_ref(reference, pci, pca, cpl)  # [B]
            ref_r = sequence_logprob_ref(reference, pri, pra, rpl)  # [B]

            loss, acc = dpo_loss(BETA, pi_c, pi_r, ref_c, ref_r)

            # Guard: ensure graph is live
            if not loss.requires_grad:
                raise RuntimeError("DPO loss is not connected to the policy graph (no gradients).")

            (loss / GA).backward()
            running += loss.item()
            running_acc += acc.item()

            if (it + 1) % GA == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1

                if step % 10 == 0:
                    avg_loss = running / 10
                    avg_acc  = running_acc / 10
                    print(f"[E{epoch+1}] step {step:5d}/{total_steps} | loss {avg_loss:.4f} | pref_acc {avg_acc:.3f}")
                    running = 0.0
                    running_acc = 0.0

    # 7) Save adapter + tokenizer
    os.makedirs(OUT_DIR, exist_ok=True)
    policy.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"[OK] DPO LoRA saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
