# code/eval/eval_slm.py

import os
import sys
import argparse
import yaml
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(THIS_DIR)
sys.path.append(CODE_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ==========================
#   BASELINE (NO LoRA)
# ==========================
def eval_slm_baseline(cfg_slm):
    slm_dir = cfg_slm["slm_baseline_dir"]
    prompt = cfg_slm["prompt"]
    max_new_tokens = int(cfg_slm.get("max_new_tokens", 128))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[eval_slm] Loading baseline SLM from: {slm_dir}")
    tokenizer = AutoTokenizer.from_pretrained(slm_dir)
    model = AutoModelForCausalLM.from_pretrained(slm_dir).to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.95,
        )

    print("\n[eval_slm] === BASELINE SLM OUTPUT ===")
    print(tokenizer.decode(out[0], skip_special_tokens=True))


# ==========================
#   LoRA (PER-CLUSTER)
# ==========================
def eval_slm_lora_clusters(cfg_slm):
    slm_lora_dir = cfg_slm["slm_lora_dir"]
    base_model_id = cfg_slm["base_model_id"]
    num_clusters = int(cfg_slm.get("num_clusters", 3))
    prompt = cfg_slm["prompt"]
    max_new_tokens = int(cfg_slm.get("max_new_tokens", 128))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[eval_slm] Base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # Load the base model ONCE
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id).to(device)
    base_model.eval()

    # Evaluate each cluster separately
    for cid in range(num_clusters):
        cluster_dir = os.path.join(slm_lora_dir, f"cluster:{cid}")

        if not os.path.isdir(cluster_dir):
            print(f"\n[eval_slm] Cluster {cid}: directory not found → {cluster_dir}")
            continue

        print(f"\n[eval_slm] ===== Cluster {cid} → loading adapter: {cluster_dir} =====")

        # Load LoRA adapter for THIS cluster
        model = PeftModel.from_pretrained(base_model, cluster_dir).to(device)
        model.eval()

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.95,
            )

        print(tokenizer.decode(out[0], skip_special_tokens=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="eval/eval_config.yaml",
        help="Eval config path (relative to code/).",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    slm_cfg = cfg["eval"]["slm"]

    if not slm_cfg.get("enabled", True):
        print("[eval_slm] SLM evaluation disabled in config.")
        return

    mode = slm_cfg.get("mode", "baseline").lower()

    if mode == "baseline":
        eval_slm_baseline(slm_cfg)
    elif mode == "lora_clusters":
        eval_slm_lora_clusters(slm_cfg)
    else:
        raise ValueError(f"Unknown slm.mode in config: {mode}")


if __name__ == "__main__":
    main()
