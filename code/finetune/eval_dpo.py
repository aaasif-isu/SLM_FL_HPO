#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------------------------
# IO helpers
# -------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def get_pairs(rows: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    pairs = []
    for r in rows:
        if "prompt" in r:
            pr = r["prompt"]
        elif "messages" in r and r["messages"] and r["messages"][0].get("role") == "user":
            pr = r["messages"][0]["content"]
        else:
            continue
        ch, rj = r.get("chosen"), r.get("rejected")
        if isinstance(pr, str) and isinstance(ch, str) and isinstance(rj, str):
            pairs.append((pr, ch, rj))
    return pairs

# -------------------------
# Model loading
# -------------------------
def load_base(model_id: str, dtype, device_map="auto"):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=device_map, trust_remote_code=True
    )
    model.eval()
    return model

def load_lora(model_id: str, lora_dir: str, dtype, device_map="auto"):
    base = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=device_map, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, lora_dir)
    model.eval()
    return model

# -------------------------
# Scoring
# -------------------------
@torch.no_grad()
def sum_logprob_of_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    device: torch.device,
) -> float:
    """
    Sum of token log-probs of `completion` given `prompt` using teacher forcing.
    Uses chat template with add_generation_prompt=True (adds assistant header).
    """

    # 1) Prompt ids via chat template
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # `apply_chat_template` may return a Tensor directly
    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids_tensor = prompt_ids
    else:
        # handle possible BatchEncoding-style return
        prompt_ids_tensor = getattr(prompt_ids, "input_ids", prompt_ids["input_ids"])

    # 2) Completion ids
    comp_enc = tokenizer(
        completion, add_special_tokens=False, return_tensors="pt"
    )
    comp_ids_tensor = comp_enc["input_ids"]

    # 3) Concatenate and move to device
    input_ids = torch.cat([prompt_ids_tensor, comp_ids_tensor], dim=1).to(device)
    attn_mask = torch.ones_like(input_ids, device=device)

    # 4) Forward pass
    out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits[:, :-1, :]         # predict next token
    targets = input_ids[:, 1:]             # shifted labels

    # 5) Slice completion region
    start = prompt_ids_tensor.size(1)      # first completion token index
    Tcomp = comp_ids_tensor.size(1)

    comp_logits = logits[:, start-1:start-1+Tcomp, :]  # [1, Tcomp, V]
    comp_tgts   = targets[:, start-1:start-1+Tcomp]    # [1, Tcomp]

    logprobs = torch.log_softmax(comp_logits, dim=-1)
    tok_lp   = logprobs.gather(2, comp_tgts.unsqueeze(-1)).squeeze(-1)  # [1, Tcomp]
    return tok_lp.sum().item()

def dpo_loss_from_margins(margins: List[float], beta: float = 0.1) -> float:
    x = torch.tensor(margins)
    # mean( -log(sigmoid(beta * margin)) ) = softplus(-beta * margin)
    return torch.nn.functional.softplus(-beta * x).mean().item()

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate preference accuracy & DPO loss on DPO pairs.")
    ap.add_argument("-d", "--data", default="/u/aalasif/SLM_FL_HPO/code/parse/dpo_flat.jsonl",
                    help="Path to DPO JSONL (flat or chat-style).")
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="Base model id.")
    ap.add_argument("--lora", default="/u/aalasif/SLM_FL_HPO/code/finetune/qwen_dpo_lora",
                    help="LoRA adapter directory.")
    ap.add_argument("--beta", type=float, default=0.1,
                    help="Beta for DPO loss computation.")
    ap.add_argument("--max_pairs", type=int, default=0,
                    help="Optional cap for quick eval.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Data
    rows = load_jsonl(Path(args.data))
    pairs = get_pairs(rows)
    if args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]
    print(f"[INFO] Loaded {len(pairs)} evaluation pairs.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Models
    base_model = load_base(args.base, dtype=dtype)
    ft_model   = load_lora(args.base, args.lora, dtype=dtype)

    # Eval helper
    @torch.no_grad()
    def eval_policy(model, name: str):
        correct = 0
        chosen_scores, rejected_scores = [], []
        for i, (prompt, chosen, rejected) in enumerate(pairs, 1):
            lp_c = sum_logprob_of_completion(model, tokenizer, prompt, chosen, device)
            lp_r = sum_logprob_of_completion(model, tokenizer, prompt, rejected, device)
            chosen_scores.append(lp_c)
            rejected_scores.append(lp_r)
            if lp_c > lp_r:
                correct += 1
            if i % 20 == 0:
                print(f"[{name}] ...scored {i}/{len(pairs)}")
        acc = correct / len(pairs) if pairs else 0.0
        return acc, chosen_scores, rejected_scores

    print("[INFO] Scoring base model (policy-only accuracy)...")
    base_acc, base_c, base_r = eval_policy(base_model, "BASE")

    print("[INFO] Scoring fine-tuned (LoRA) model (policy-only accuracy)...")
    ft_acc, ft_c, ft_r = eval_policy(ft_model, "FT")

    # DPO margin & loss (FT vs Base as reference)
    margins = [(fc - fr) - (bc - br) for fc, fr, bc, br in zip(ft_c, ft_r, base_c, base_r)]
    dpo_loss = dpo_loss_from_margins(margins, beta=args.beta)

    def mean(xs): return float(sum(xs)/len(xs)) if xs else float("nan")

    print("\n================ DPO EVAL REPORT ================")
    print(f"#pairs: {len(pairs)}")
    print(f"Base policy-only accuracy:     {base_acc*100:.2f}%")
    print(f"Fine-tuned policy-only acc:    {ft_acc*100:.2f}%")
    print(f"Mean policy margin (base):     {mean([c-r for c,r in zip(base_c, base_r)]):.4f}")
    print(f"Mean policy margin (finetune): {mean([c-r for c,r in zip(ft_c, ft_r)]):.4f}")
    print(f"DPO loss (FT vs base ref, β={args.beta}): {dpo_loss:.4f}")
    print("=================================================\n")

if __name__ == "__main__":
    main()
