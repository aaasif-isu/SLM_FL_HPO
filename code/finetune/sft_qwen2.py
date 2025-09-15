import os, json, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# ----------------------------
# Paths
# ----------------------------
MODEL_ID  = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_PATH = os.environ.get("DATA_PATH", "/u/aalasif/SLM_FL_HPO/code/parse/sft_instruct.jsonl")
OUT_DIR   = os.environ.get("OUT_DIR",  "/u/aalasif/SLM_FL_HPO/code/finetune/qwen_sft_lora")

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
# Robust JSONL reader (handles pretty/multiline)
# ----------------------------
def iter_jsonl(path):
    buf = ""
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            buf = line if not buf else (buf + "\n" + line)
            try:
                obj = json.loads(buf)
                yield obj
                buf = ""
            except json.JSONDecodeError:
                continue
        if buf.strip():
            # trailing
            obj = json.loads(buf)
            yield obj

# ----------------------------
# Load instruction-style and convert to chat
# ----------------------------
def to_str(x):
    return x if isinstance(x, str) else json.dumps(x, separators=(",", ":"), ensure_ascii=False)

rows = []
for ex in iter_jsonl(DATA_PATH):
    instr = to_str(ex.get("instruction", ""))
    inp   = to_str(ex.get("input", "")).strip()
    outp  = ex.get("output", "")

    # output should be a JSON string target; if not string, stringify
    label_json = outp if isinstance(outp, str) else json.dumps(outp, separators=(",", ":"), ensure_ascii=False)

    user_content = instr if not inp else (instr + "\n" + inp)

    messages = [
        {
            "role": "system",
            "content": "You are a hyperparameter assistant. Output ONLY JSON with fields {client, server, mu}; no extra text."
        },
        {"role": "user", "content": user_content},
    ]

    rows.append({"messages": messages, "label_json": label_json, "metadata": ex.get("metadata", {})})

if not rows:
    raise ValueError(f"No training examples found in {DATA_PATH}.")

ds = Dataset.from_list(rows)

# ----------------------------
# Tokenizer / Model
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if USE_4BIT else None

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_cfg
)

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_cfg)

# ----------------------------
# Build train strings: chat template + label_json + </s>
# ----------------------------
def build_row(ex):
    text = tokenizer.apply_chat_template(
        ex["messages"],
        tokenize=False,
        add_generation_prompt=True  # appends assistant prefix
    )
    full = text + ex["label_json"] + tokenizer.eos_token
    return {"full_text": full, "prompt_text": text, "label_json": ex["label_json"]}

ds_proc = ds.map(build_row, remove_columns=ds.column_names)

# Mask loss on the prompt; compute loss only over the assistant JSON
def tokenize(ex):
    full = tokenizer(ex["full_text"], max_length=MAX_LEN, truncation=True)
    prompt = tokenizer(ex["prompt_text"], max_length=MAX_LEN, truncation=True)

    input_ids = full["input_ids"]
    labels = input_ids[:]  # copy
    # mask prompt
    cut = min(len(prompt["input_ids"]), len(labels))
    for i in range(cut):
        labels[i] = -100
    return {"input_ids": input_ids, "labels": labels, "attention_mask": full["attention_mask"]}

tok_ds = ds_proc.map(tokenize, remove_columns=ds_proc.column_names)

class DataCollatorForCausalJSON:
    def __init__(self, tokenizer, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_masks, labels = [], [], []
        for f in features:
            ids, att, lab = f["input_ids"], f["attention_mask"], f["labels"]
            pad = max_len - len(ids)
            if pad > 0:
                ids = ids + [pad_id] * pad
                att = att + [0] * pad
                lab = lab + [self.label_pad_token_id] * pad
            input_ids.append(ids)
            attention_masks.append(att)
            labels.append(lab)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

collator = DataCollatorForCausalJSON(tokenizer)

# ----------------------------
# Train
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
    report_to=[],
)

trainer = Trainer(model=model, args=args, train_dataset=tok_ds, data_collator=collator)
trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"SFT done. LoRA saved to: {OUT_DIR}")

# ----------------------------
# Optional: evaluate base vs fine-tuned (quantized to avoid OOM)
# ----------------------------
try:
    from eval_loss import evaluate_loss
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, device_map="auto",
        quantization_config=bnb_cfg
    )
    base.eval(); model.eval()
    base_loss = evaluate_loss(base, tok_ds, batch_size=BATCH, collate_fn=collator)
    ft_loss   = evaluate_loss(model, tok_ds, batch_size=BATCH, collate_fn=collator)
    print(f"Base model loss on dataset:     {base_loss:.4f}")
    print(f"Fine-tuned (LoRA) loss on data: {ft_loss:.4f}")
except Exception as e:
    print("Skipping eval (optional):", e)
