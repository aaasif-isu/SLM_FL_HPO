import os, json, torch, re
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import List, Dict
from eval_loss import evaluate_loss




# ----------------------------
# Paths (no CLI args needed)
# ----------------------------
MODEL_ID  = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dummy_sft.jsonl")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "qwen_sft_lora")

# ----------------------------
# Train hyperparams
# ----------------------------
USE_4BIT = bool(int(os.environ.get("USE_4BIT", "1")))   # 1 = enable 4-bit
EPOCHS   = float(os.environ.get("EPOCHS", "2"))
LR       = float(os.environ.get("LR", "5e-5"))
BATCH    = int(os.environ.get("BATCH", "2"))
GA       = int(os.environ.get("GA", "8"))
MAX_LEN  = int(os.environ.get("MAX_LEN", "1536"))

# ----------------------------
# Helpers
# ----------------------------
def _to_str(x):
    return x if isinstance(x, str) else json.dumps(x, separators=(",", ":"))

def _iter_json_objects_allow_multiline(path):
    """
    Robust reader: yields JSON objects from a file that may contain:
      - true JSONL (one object per line), or
      - pretty-printed multi-line JSON objects back-to-back.
    Strategy: accumulate lines until we can json.loads(buffer) successfully.
    """
    buf = ""
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # allow comments starting with //
            if line.startswith("//"):
                continue

            if not buf:
                buf = line
            else:
                buf += "\n" + line

            # Try parse current buffer. If it fails, keep accumulating.
            try:
                obj = json.loads(buf)
                yield obj
                buf = ""  # reset after successful parse
            except json.JSONDecodeError:
                # not complete yet, continue accumulating
                continue

        # trailing buffer check
        if buf.strip():
            try:
                obj = json.loads(buf)
                yield obj
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Could not parse trailing JSON object. "
                    f"Make sure there are no trailing commas and the file is valid JSON."
                ) from e

# ----------------------------
# 1) Load & normalize examples
#    - Supports multi-line records.
#    - Ensures messages[].content is ALWAYS a string.
#    - Inserts a default system message if missing.
# ----------------------------
rows = []
for ex in _iter_json_objects_allow_multiline(DATA_PATH):
    msgs = ex.get("messages", [])
    # Insert a default system message if none present
    if not any(m.get("role") == "system" for m in msgs):
        msgs = [{
            "role": "system",
            "content": "You are a hyperparameter assistant. Output ONLY JSON with keys {lr,batch,wd,optimizer,scheduler,dropout}."
        }] + msgs

    # Normalize roles and content
    for m in msgs:
        m["role"] = str(m.get("role", "user"))
        m["content"] = _to_str(m.get("content", ""))  # stringify dicts/lists

    ex["messages"] = msgs

    # Label must be a dict
    if not isinstance(ex.get("label"), dict):
        raise ValueError("Each example must have a dict under key 'label' (the HP JSON).")

    rows.append(ex)

if not rows:
    raise ValueError(f"No training examples found in {DATA_PATH}.")

ds = Dataset.from_list(rows)

# ----------------------------
# 2) Tokenizer / Model
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_cfg = None
if USE_4BIT:
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_cfg
)

# LoRA config
peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)

# ----------------------------
# 3) Build training strings
#    chat_template(messages) + assistant JSON + </s>
# ----------------------------
def build_row(ex):
    messages = ex["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # puts assistant header at end
    )
    label_json = json.dumps(ex["label"], separators=(",", ":"), ensure_ascii=False)
    full = text + label_json + tokenizer.eos_token
    return {"full_text": full, "prompt_text": text, "label_json": label_json}

ds_proc = ds.map(build_row, remove_columns=ds.column_names)

# Mask loss on the prompt; compute loss only over the assistant JSON
def tokenize(ex):
    full_ids   = tokenizer(ex["full_text"], max_length=MAX_LEN, truncation=True)
    prompt_ids = tokenizer(ex["prompt_text"], max_length=MAX_LEN, truncation=True)

    input_ids = full_ids["input_ids"]
    labels    = input_ids[:]  # copy

    # mask prompt tokens with -100
    cut = min(len(prompt_ids["input_ids"]), len(labels))
    for i in range(cut):
        labels[i] = -100

    return {"input_ids": input_ids, "labels": labels, "attention_mask": full_ids["attention_mask"]}

tok_ds = ds_proc.map(tokenize, remove_columns=ds_proc.column_names)

class DataCollatorForCausalJSON:
    def __init__(self, tokenizer, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids, attention_masks, labels = [], [], []
        for f in features:
            ids = f["input_ids"]
            att = f["attention_mask"]
            lab = f["labels"]

            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.tokenizer.pad_token_id] * pad_len
                att = att + [0] * pad_len
                lab = lab + [self.label_pad_token_id] * pad_len

            input_ids.append(ids)
            attention_masks.append(att)
            labels.append(lab)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


collator = DataCollatorForCausalJSON(tokenizer=tokenizer, label_pad_token_id=-100)

# ----------------------------
# 4) Train
# ----------------------------
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GA,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    bf16=True,
    optim="adamw_torch",
)

trainer = Trainer(model=model, args=args, train_dataset=tok_ds, data_collator=collator)
trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"SFT done. LoRA saved to: {OUT_DIR}")

# Evaluate base vs fine-tuned

# Fresh base (no LoRA)
base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, device_map="auto"
)

# Use the SAME collator so eval batches are padded the same way
base_loss = evaluate_loss(base, tok_ds, batch_size=BATCH, collate_fn=collator)
ft_loss   = evaluate_loss(model, tok_ds, batch_size=BATCH, collate_fn=collator)

print(f"Base model loss on dataset:     {base_loss:.4f}")
print(f"Fine-tuned (LoRA) loss on data: {ft_loss:.4f}")
