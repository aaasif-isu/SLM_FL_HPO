#!/usr/bin/env python3
import json
import re
import argparse
from typing import Optional, Dict, Any, List

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
                    out[-1] = "\\"
                    out.append("\\")
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

# ----------------------- Patterns -----------------------

EPOCH_RE   = re.compile(r"===\s*Global\s*Epoch\s*(\d+)\s*/\s*\d+\s*===", re.IGNORECASE)
CLUSTER_RE = re.compile(r"\*\*\*Cluster\s+(\d+)\s+\(FL Mode:\s*([A-Za-z0-9_\-]+)\)\s+with\s+members\s+\[([^\]]+)\]\*\*\*", re.IGNORECASE)

# Old split header (still useful, but not sufficient on its own)
CLIENT_SPLIT_RE = re.compile(r"-->\s*Client\s+(\d+):", re.IGNORECASE)

# Capture ANY HP Agent JSON block, even if header wraps across lines
HP_AGENT_ANY_RE = re.compile(
    r"<<<\s*RESPONSE\s*FROM\s*HP\s*AGENT\s*\(Client\s*(\d+)\)\s*:\s*(\{.*?\})\s*-{10,}",
    re.IGNORECASE | re.DOTALL
)

# Analyzer blocks (optional, stored as events)
ANALYZER_ANY_RE = re.compile(
    r"<<<\s*RESPONSE\s*FROM\s*ANALYZER\s*AGENT\s*\(Client\s*(\d+)\)\s*:\s*(\{.*?\})\s*-{10,}",
    re.IGNORECASE | re.DOTALL
)

# Accuracy lines anywhere in the cluster/epoch
ACC_RE = re.compile(r"Client\s+(\d+),\s*Local\s*Epochs\s*\d+:\s*Train\s*Acc\s*([\d.]+)%\s*,\s*Test\s*Acc\s*([\d.]+)%",
                    re.IGNORECASE)

# ---- meta patterns ----
DATASET_RE      = re.compile(r"\bDataset:\s*([A-Za-z0-9_\-]+)", re.IGNORECASE)
NONIID_RE       = re.compile(r"\bNon-IIDness:\s*([0-9.]+)", re.IGNORECASE)
MODEL_LAYERS_RE = re.compile(r"Total layer[s]?\s+in\s+([A-Za-z0-9_\-]+)\s+is", re.IGNORECASE)
SAVE_PATH_RE    = re.compile(
    r"hpo_state_([A-Za-z0-9_\-]+)__([A-Za-z0-9_\-]+)_clients(\d+)_imb([0-9.]+)_epochs(\d+)_splitfed",
    re.IGNORECASE
)
CLIENTS_RE      = re.compile(r"Training with\s+(\d+)\s+clients\.", re.IGNORECASE)
FL_MODE_LINE_RE = re.compile(r"---\s*Using\s*FL\s*Mode:\s*([A-Za-z0-9_\-]+)\s*---", re.IGNORECASE)
HPO_STRAT_RE    = re.compile(r"---\s*Using\s*HPO\s*Strategy:\s*([A-Za-z0-9_\-]+)\s*---", re.IGNORECASE)

CONTEXT_TEMPLATE = (
    "Suggest hyperparameters for the next local training round in federated split learning.\n"
    "Context: dataset={dataset}; fl_mode={fl_mode}; model={model}; client_profile=low.\n"
    "Return ONLY JSON with fields {{client, server, mu}}; no extra text."
)

# ----------------------- Meta extraction -----------------------

def extract_run_meta(log_content: str) -> dict:
    meta = {}

    m = SAVE_PATH_RE.search(log_content)
    if m:
        meta["model"] = m.group(1)
        meta["dataset"] = m.group(2)
        meta["num_clients"] = int(m.group(3))
        meta["imbalance_ratio"] = float(m.group(4))
        meta["total_epochs"] = int(m.group(5))

    if "dataset" not in meta:
        m = DATASET_RE.search(log_content)
        if m:
            meta["dataset"] = m.group(1)

    if "model" not in meta:
        m = MODEL_LAYERS_RE.search(log_content)
        if m:
            meta["model"] = m.group(1)

    if "imbalance_ratio" not in meta:
        m = NONIID_RE.search(log_content)
        if m:
            try:
                meta["imbalance_ratio"] = float(m.group(1))
            except ValueError:
                pass

    m = CLIENTS_RE.search(log_content)
    if m and "num_clients" not in meta:
        meta["num_clients"] = int(m.group(1))

    m = FL_MODE_LINE_RE.search(log_content)
    meta["fl_mode"] = m.group(1) if m else "SPLITFED"

    m = HPO_STRAT_RE.search(log_content)
    if m:
        meta["hpo_strategy"] = m.group(1)

    if "dataset" in meta:
        meta["dataset"] = meta["dataset"].lower()

    return meta

# ----------------------- Parser -----------------------

def ensure_client(clients_idx: Dict[int, Dict[str, Any]], cluster_clients: List[Dict[str, Any]],
                  client_id: int, context_text: str) -> Dict[str, Any]:
    """Create (or return) a client object inside this cluster."""
    if client_id not in clients_idx:
        obj = {
            "client_id": client_id,
            "context": context_text,
            "hps": {},                 # last suggested (if any)
            "hp_events": [],           # list of all suggested hps (order preserved)
            "analyzer_events": [],     # list of analyzer JSONs (optional)
        }
        clients_idx[client_id] = obj
        cluster_clients.append(obj)
    return clients_idx[client_id]

def parse_log_to_json(log_content: str) -> dict:
    meta = extract_run_meta(log_content)
    parsed = {"meta": meta, "epochs": []}

    context_text = CONTEXT_TEMPLATE.format(
        dataset=meta.get("dataset", "cifar10"),
        fl_mode=meta.get("fl_mode", "SPLITFED"),
        model=meta.get("model", "ResNet18"),
    )

    # Split by epochs
    parts = re.split(EPOCH_RE, log_content)
    if len(parts) <= 1:
        # No epoch markers; treat whole log as one epoch 1
        parts = ["", "1", log_content]

    # parts looks like: [prefix, epoch1_num, epoch1_body, epoch2_num, epoch2_body, ...]
    # start from index 1
    for i in range(1, len(parts), 2):
        epoch_num = int(parts[i])
        epoch_body = parts[i + 1]

        epoch_obj = {"global_epoch": epoch_num, "clusters": []}

        # Split by clusters
        cparts = re.split(CLUSTER_RE, epoch_body)
        if len(cparts) == 1:
            # No explicit clusters detected; make a single cluster -1
            cluster_obj = {
                "cluster_id": -1,
                "fl_mode": meta.get("fl_mode", "SPLITFED"),
                "members": [],
                "clients": []
            }
            clients_idx = {}

            # Scan HP Agent blocks in the whole epoch
            for m in HP_AGENT_ANY_RE.finditer(epoch_body):
                cid = int(m.group(1))
                raw = m.group(2).strip()
                hp = try_json(raw) or repair_json_block(raw)
                if not hp or "hps" not in hp:
                    continue
                c = ensure_client(clients_idx, cluster_obj["clients"], cid, context_text)
                # reasoning (last seen)
                if isinstance(hp.get("reasoning"), str):
                    c["reasoning"] = " ".join(hp["reasoning"].split())
                c["hp_events"].append(hp["hps"])
                c["hps"] = hp["hps"]

            # Analyzer
            for m in ANALYZER_ANY_RE.finditer(epoch_body):
                cid = int(m.group(1))
                raw = m.group(2).strip()
                an = try_json(raw) or repair_json_block(raw)
                if not an:
                    continue
                c = ensure_client(clients_idx, cluster_obj["clients"], cid, context_text)
                c["analyzer_events"].append(an)

            # Acc lines
            for m in ACC_RE.finditer(epoch_body):
                cid = int(m.group(1))
                c = ensure_client(clients_idx, cluster_obj["clients"], cid, context_text)
                c["train_acc"] = float(m.group(2))
                c["test_acc"] = float(m.group(3))

            epoch_obj["clusters"].append(cluster_obj)
            parsed["epochs"].append(epoch_obj)
            continue

        # cparts = [prefix, cluster_id, fl_mode, members, cluster_body, cluster_id, fl_mode, members, cluster_body, ...]
        # walk groups of 4 after the first prefix
        for j in range(1, len(cparts), 4):
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
            clients_idx: Dict[int, Dict[str, Any]] = {}

            # PASS A: capture all HP Agent blocks in the cluster (even if no arrow header)
            for m in HP_AGENT_ANY_RE.finditer(cluster_body):
                cid = int(m.group(1))
                raw = m.group(2).strip()
                hp = try_json(raw) or repair_json_block(raw)
                if not hp or "hps" not in hp:
                    continue
                c = ensure_client(clients_idx, cluster_obj["clients"], cid, context_text)
                if isinstance(hp.get("reasoning"), str):
                    c["reasoning"] = " ".join(hp["reasoning"].split())
                c["hp_events"].append(hp["hps"])
                c["hps"] = hp["hps"]

            # PASS B: Analyzer blocks (optional diagnostics)
            for m in ANALYZER_ANY_RE.finditer(cluster_body):
                cid = int(m.group(1))
                raw = m.group(2).strip()
                an = try_json(raw) or repair_json_block(raw)
                if not an:
                    continue
                c = ensure_client(clients_idx, cluster_obj["clients"], cid, context_text)
                c["analyzer_events"].append(an)

            # PASS C: Accuracy lines (ensure we still capture clients even without HP blocks)
            for m in ACC_RE.finditer(cluster_body):
                cid = int(m.group(1))
                c = ensure_client(clients_idx, cluster_obj["clients"], cid, context_text)
                c["train_acc"] = float(m.group(2))
                c["test_acc"] = float(m.group(3))

            # (Optional) PASS D: legacy client split (if you want to harvest anything else living only inside)
            # This can catch exotic formats where JSON appears but our ANY_RE didn't match.
            kparts = re.split(CLIENT_SPLIT_RE, cluster_body)[1:]
            for k in range(0, len(kparts), 2):
                try:
                    cid = int(kparts[k])
                except Exception:
                    continue
                client_body = kparts[k + 1]
                # Try to find any JSON in this slice if we didn't already
                if cid not in clients_idx:
                    # Try a last-ditch search for a JSON block following this header
                    mjson = re.search(r"(\{.*?\})\s*-{10,}", client_body, re.DOTALL)
                    if mjson:
                        hp = try_json(mjson.group(1)) or repair_json_block(mjson.group(1))
                        if hp and "hps" in hp:
                            c = ensure_client(clients_idx, cluster_obj["clients"], cid, context_text)
                            if isinstance(hp.get("reasoning"), str):
                                c["reasoning"] = " ".join(hp["reasoning"].split())
                            c["hp_events"].append(hp["hps"])
                            c["hps"] = hp["hps"]
                # Accuracy inside this slice (kept for completeness)
                for am in ACC_RE.finditer(client_body):
                    cid2 = int(am.group(1))
                    c2 = ensure_client(clients_idx, cluster_obj["clients"], cid2, context_text)
                    c2["train_acc"] = float(am.group(2))
                    c2["test_acc"] = float(am.group(3))

            epoch_obj["clusters"].append(cluster_obj)

        parsed["epochs"].append(epoch_obj)

    return parsed



# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser(description="Parse FL log into JSON structure (captures all client appearances).")
    ap.add_argument("-i", "--input",  default="/u/aalasif/SLM_FL_HPO/code/parse/main_data.log", help="Path to log file")
    ap.add_argument("-o", "--output", default="/u/aalasif/SLM_FL_HPO/code/parse/epoch_client_hps.json", help="Path to write JSON output")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    data = parse_log_to_json(content)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()
