import json
import re
import argparse
from typing import Optional

# ----------------------- Robust JSON repair helpers -----------------------

SMARTS = {
    "\u201c": '"', "\u201d": '"',   # curly double quotes
    "\u2018": "'", "\u2019": "'",   # curly single quotes
    "\u2013": "-",  "\u2014": "-",  # en/em dash
}
VALID_ESCAPES = set(list("btnfr\"\\/"))  # valid JSON escapes after backslash

def replace_smart_punct(s: str) -> str:
    for k, v in SMARTS.items():
        s = s.replace(k, v)
    return s

def strip_control_chars(s: str) -> str:
    # Keep \n and \t; drop other control chars
    return "".join(ch for ch in s if ch >= " " or ch in "\n\t")

def remove_trailing_commas(s: str) -> str:
    # Remove trailing commas before } or ]
    return re.sub(r",\s*([}\]])", r"\1", s)

def balance_braces(s: str) -> Optional[str]:
    """Trim to last balanced closing brace starting at first '{'."""
    first = s.find("{")
    if first == -1:
        return None
    s = s[first:]
    depth, last_good = 0, -1
    for i, ch in enumerate(s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_good = i
    return s[:last_good + 1] if last_good != -1 else None

def escape_bad_backslashes(s: str) -> str:
    out, in_str, escape, quote = [], False, False, None
    i = 0
    while i < len(s):
        ch = s[i]
        if not in_str:
            if ch in ('"', "'"):
                in_str, quote = True, ch
            out.append(ch)
            i += 1
        else:
            if escape:
                if ch not in VALID_ESCAPES:
                    out[-1] = "\\"  # keep previous backslash
                    out.append("\\")  # add extra to make it escaped
                out.append(ch)
                escape = False
                i += 1
            else:
                if ch == "\\":
                    out.append(ch)
                    escape = True
                    i += 1
                elif ch == quote:
                    in_str, quote = False, None
                    out.append(ch)
                    i += 1
                else:
                    out.append(ch)
                    i += 1
    if escape:
        out.append("\\")
    return "".join(out)

def try_json(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None

def repair_json_block(raw: str):
    s = replace_smart_punct(raw)
    s = strip_control_chars(s)
    s = balance_braces(s) or s
    s = remove_trailing_commas(s)
    s = escape_bad_backslashes(s)
    return try_json(s)

# ----------------------- Core parsing -----------------------

EPOCH_RE   = re.compile(r"=== Global Epoch (\d+)/\d+ ===")
CLUSTER_RE = re.compile(r"\*\*\*Cluster (\d+) \(FL Mode: (SPLITFED)\) with members \[([^\]]+)\]\*\*\*")
CLIENT_SPLIT_RE = re.compile(r"-->\s*Client\s+(\d+):")
HP_AGENT_BLOCK_RE = re.compile(
    r"<<< RESPONSE FROM HP AGENT.*?:\s*(\{.*?\})\s*-{10,}",
    re.DOTALL | re.IGNORECASE
)
ACC_RE = re.compile(r"Client (\d+), Local Epochs \d+: Train Acc ([\d.]+)%, Test Acc ([\d.]+)%")

# ---- NEW: metadata patterns ----
DATASET_RE      = re.compile(r"\bDataset:\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)
NONIID_RE       = re.compile(r"\bNon-IIDness:\s*([0-9.]+)", re.IGNORECASE)
MODEL_LAYERS_RE = re.compile(r"Total layer[s]?\s+in\s+([A-Za-z0-9_\-]+)\s+is", re.IGNORECASE)
SAVE_PATH_RE    = re.compile(
    r"hpo_state_([A-Za-z0-9_\-]+)__([A-Za-z0-9_\-]+)_clients(\d+)_imb([0-9.]+)_epochs(\d+)_splitfed",
    re.IGNORECASE
)
CLIENTS_RE      = re.compile(r"Training with\s+(\d+)\s+clients\.", re.IGNORECASE)
FL_MODE_LINE_RE = re.compile(r"---\s*Using FL Mode:\s*([A-Za-z0-9_\-]+)\s*---", re.IGNORECASE)
HPO_STRAT_RE    = re.compile(r"---\s*Using HPO Strategy:\s*([A-Za-z0-9_\-]+)\s*---", re.IGNORECASE)

CONTEXT_TEMPLATE = (
    "Suggest hyperparameters for the next local training round in federated split learning.\n"
    "Context: dataset={dataset}; fl_mode={fl_mode}; model={model}; client_profile=low.\n"
    "Return ONLY JSON with fields {{client, server, mu}}; no extra text."
)

def extract_run_meta(log_content: str) -> dict:
    meta = {}

    # Prefer extracting from the saved-state path if present
    m = SAVE_PATH_RE.search(log_content)
    if m:
        meta["model"] = m.group(1)
        meta["dataset"] = m.group(2)
        meta["num_clients"] = int(m.group(3))
        meta["imbalance_ratio"] = float(m.group(4))
        meta["total_epochs"] = int(m.group(5))

    # Dataset fallback(s)
    if "dataset" not in meta:
        m = DATASET_RE.search(log_content)
        if m:
            meta["dataset"] = m.group(1)

    # Model fallback
    if "model" not in meta:
        m = MODEL_LAYERS_RE.search(log_content)
        if m:
            meta["model"] = m.group(1)

    # Imbalance / non-IIDness fallback
    if "imbalance_ratio" not in meta:
        m = NONIID_RE.search(log_content)
        if m:
            try:
                meta["imbalance_ratio"] = float(m.group(1))
            except ValueError:
                pass

    # Misc extras (optional)
    m = CLIENTS_RE.search(log_content)
    if m and "num_clients" not in meta:
        meta["num_clients"] = int(m.group(1))

    m = FL_MODE_LINE_RE.search(log_content)
    if m:
        meta["fl_mode"] = m.group(1)
    else:
        # default if not found
        meta["fl_mode"] = "SPLITFED"

    m = HPO_STRAT_RE.search(log_content)
    if m:
        meta["hpo_strategy"] = m.group(1)

    # Normalize casing a bit
    if "dataset" in meta:
        meta["dataset"] = meta["dataset"].lower()
    if "model" in meta:
        meta["model"] = meta["model"]

    return meta

def parse_log_to_json(log_content: str) -> dict:
    meta = extract_run_meta(log_content)

    parsed = {"meta": meta, "epochs": []}

    # Build the context-aware prompt once
    context_text = CONTEXT_TEMPLATE.format(
        dataset=meta.get("dataset", "cifar10"),
        fl_mode=meta.get("fl_mode", "SPLITFED"),
        model=meta.get("model", "ResNet18"),
    )

    # Split by epochs (re.split keeps captured group values)
    parts = re.split(EPOCH_RE, log_content)[1:]
    # parts = [epoch_num, epoch_body, epoch_num, epoch_body, ...]
    for i in range(0, len(parts), 2):
        epoch_num = int(parts[i])
        epoch_body = parts[i + 1]

        epoch_obj = {"global_epoch": epoch_num, "clusters": []}

        # Split by clusters; regex has 3 groups -> id, mode, members + body
        cparts = re.split(CLUSTER_RE, epoch_body)[1:]
        # cparts = [cluster_id, fl_mode, members_str, cluster_body, ...]
        for j in range(0, len(cparts), 4):
            cluster_id = int(cparts[j])
            fl_mode = cparts[j + 1]
            members_str = cparts[j + 2]
            cluster_body = cparts[j + 3]

            cluster_obj = {
                "cluster_id": cluster_id,
                "fl_mode": fl_mode,
                "members": [int(x.strip()) for x in members_str.split(",") if x.strip()],
                "clients": []
            }

            # Split cluster body by clients
            kparts = re.split(CLIENT_SPLIT_RE, cluster_body)[1:]
            # kparts = [client_id, client_body, client_id, client_body, ...]
            for k in range(0, len(kparts), 2):
                client_id = int(kparts[k])
                client_body = kparts[k + 1]

                # Find first HP AGENT block after this client line
                hp_match = HP_AGENT_BLOCK_RE.search(client_body)
                chosen_hps = {}
                reasoning_text = None

                if hp_match:
                    raw_json = hp_match.group(1).strip()
                    # Try to parse or repair
                    hp = try_json(raw_json) or repair_json_block(raw_json)

                    if hp:
                        # NEW: parse reasoning
                        if isinstance(hp.get("reasoning"), str):
                            # optional: collapse excessive whitespace
                            reasoning_text = " ".join(hp["reasoning"].split())
                    if hp and "hps" in hp:
                        chosen_hps = hp["hps"]

                # Optional: nearest accuracy for this client within this client block
                train_acc = test_acc = None
                for m in ACC_RE.finditer(client_body):
                    if int(m.group(1)) == client_id:
                        train_acc = float(m.group(2))
                        test_acc = float(m.group(3))
                        break

                client_obj = {
                    "client_id": client_id,
                    "context": context_text,
                    "hps": chosen_hps
                }

                if reasoning_text:
                    client_obj["reasoning"] = reasoning_text
                if train_acc is not None:
                    client_obj["train_acc"] = train_acc
                if test_acc is not None:
                    client_obj["test_acc"] = test_acc

                cluster_obj["clients"].append(client_obj)

            epoch_obj["clusters"].append(cluster_obj)

        parsed["epochs"].append(epoch_obj)

    return parsed

def main():
    ap = argparse.ArgumentParser(description="Parse FL log into JSON structure.")
    ap.add_argument("-i", "--input",  default="/u/aalasif/SLM_FL_HPO/code/parse/4_clients_run.log", help="Path to log file")
    ap.add_argument("-o", "--output", default="/u/aalasif/SLM_FL_HPO/code/parse/4_clients_run.json", help="Path to write JSON output")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    data = parse_log_to_json(content)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()
