#!/usr/bin/env python3
import os, json, torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ----------------------------
# Config
# ----------------------------
MODEL_ID    = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "//u/aalasif/SLM_FL_HPO/code/finetune/qwen_sft_lora")
USE_4BIT    = bool(int(os.environ.get("USE_4BIT", "1")))
MAX_NEW     = int(os.environ.get("MAX_NEW", "256"))
TEMP        = float(os.environ.get("TEMP", "0.0"))
TOP_P       = float(os.environ.get("TOP_P", "1.0"))
SHOW_RAW    = bool(int(os.environ.get("SHOW_RAW_DEBUG", "0")))  # 1 to see raw text on failures

REQUIRED_KEYS = ("client", "server", "mu")

SYSTEM_PROMPT = (
    "You are a hyperparameter assistant. "
    "Return ONLY JSON with fields {client, server, mu}; no extra text."
)

def build_user_prompt(dataset="cifar10", fl_mode="SPLITFED", model_name="ResNet18", client_profile="low"):
    return (
        "Suggest hyperparameters for the next local training round in federated split learning.\n"
        f"Context: dataset={dataset}; fl_mode={fl_mode}; model={model_name}; client_profile={client_profile}.\n"
        "Return ONLY JSON with fields {client, server, mu}; no extra text."
    )

# ----------------------------
# Errors
# ----------------------------
class JSONNotFoundError(Exception): ...
class SchemaMissingError(Exception):
    def __init__(self, missing_keys):
        super().__init__(f"Missing keys: {missing_keys}")
        self.missing_keys = missing_keys

# ----------------------------
# JSON extraction & validation
# ----------------------------
def extract_json(text: str):
    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # bracket scan
    starts = [m.start() for m in re.finditer(r"\{", text)]
    ends   = [m.start() for m in re.finditer(r"\}", text)]
    for i in range(len(starts)):
        for j in range(len(ends)-1, -1, -1):
            if ends[j] >= starts[i]:
                chunk = text[starts[i]:ends[j]+1]
                try:
                    return json.loads(chunk)
                except Exception:
                    continue
    raise JSONNotFoundError("No valid JSON object found in model output.")

def find_missing_keys(obj: dict, required=REQUIRED_KEYS):
    return [k for k in required if k not in obj]

def validate_schema(obj: dict):
    if not isinstance(obj, dict):
        raise SchemaMissingError(["<root-not-object>"])
    missing = find_missing_keys(obj, REQUIRED_KEYS)
    if missing:
        raise SchemaMissingError(missing)
    return obj

# ----------------------------
# Prompt & generation
# ----------------------------
def build_prompt(tokenizer, dataset, fl_mode, model_name, client_profile):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_prompt(dataset, fl_mode, model_name, client_profile)}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def generate_with(model, tokenizer, prompt, max_new_tokens=256, temperature=0.0, top_p=1.0):
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0.0:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
    else:
        gen_kwargs.update(dict(do_sample=False))

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out_ids = model.generate(**inputs, **gen_kwargs)
    out = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    obj = extract_json(out)
    obj = validate_schema(obj)
    return out, obj

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if USE_4BIT else None

    # Base
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, device_map="auto", quantization_config=bnb_cfg
    ).eval()

    # Fine-tuned (LoRA)
    ft_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, device_map="auto", quantization_config=bnb_cfg
    )
    ft = PeftModel.from_pretrained(ft_base, ADAPTER_DIR).eval()

    # Clean generation_config knobs for greedy mode to avoid warnings
    for m in (base, ft):
        try:
            m.generation_config.top_k = None
            m.generation_config.top_p = None
            m.generation_config.temperature = None
        except Exception:
            pass

    prompt = build_prompt(
        tokenizer,
        dataset=os.environ.get("DATASET", "cifar10"),
        fl_mode=os.environ.get("FL_MODE", "SPLITFED"),
        model_name=os.environ.get("MODEL_NAME", "ResNet18"),
        client_profile=os.environ.get("CLIENT_PROFILE", "low"),
    )

    # -------- Base model
    try:
        base_raw, base_json = generate_with(base, tokenizer, prompt, MAX_NEW, TEMP, TOP_P)
        print("==== Base model parsed JSON ====")
        print(json.dumps(base_json, indent=2))
    except SchemaMissingError as e:
        print("==== Base model skipped ====")
        print(f"Reason: JSON missing required keys -> {e.missing_keys}")
        if SHOW_RAW:
            print("[Raw base output suppressed by default; enable with SHOW_RAW_DEBUG=1]\n")
    except JSONNotFoundError as e:
        print("==== Base model skipped ====")
        print("Reason: Could not find a valid JSON object in output.")
        if SHOW_RAW:
            print("[Raw base output suppressed by default; enable with SHOW_RAW_DEBUG=1]\n")
    except Exception as e:
        print("==== Base model skipped ====")
        print(f"Reason: Unexpected error -> {repr(e)}")
        if SHOW_RAW:
            print("[Raw base output suppressed by default; enable with SHOW_RAW_DEBUG=1]\n")

    # -------- Fine-tuned (LoRA)
    try:
        ft_raw, ft_json = generate_with(ft, tokenizer, prompt, MAX_NEW, TEMP, TOP_P)
        print("\n==== Fine-tuned (LoRA) parsed JSON ====")
        print(json.dumps(ft_json, indent=2))
    except Exception as e:
        print("\n==== Fine-tuned (LoRA) error ====")
        print(repr(e))
        if SHOW_RAW:
            print("[Enable SHOW_RAW_DEBUG=1 to print raw output]")
