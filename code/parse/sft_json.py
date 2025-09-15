#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List

REQ_KEYS = ("client", "server", "mu")

def to_minified_json(obj: Dict[str, Any]) -> str:
    """Minify JSON for use as the assistant target string."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

def split_instruction_input(ctx: str) -> Tuple[str, str]:
    """
    Split a multi-line context into (instruction, input).
    Instruction = first line; input = remaining lines.
    """
    parts = (ctx or "").splitlines()
    if not parts:
        return "", ""
    instr = parts[0].strip()
    rest = "\n".join(p.strip() for p in parts[1:]).strip()
    return instr, rest

def build_fallback_context(meta: dict) -> str:
    """Fallback context if a client entry lacks 'context'."""
    dataset = meta.get("dataset", "cifar10")
    fl_mode = meta.get("fl_mode", "SPLITFED")
    model = meta.get("model", "ResNet18")
    return (
        "Suggest hyperparameters for the next local training round in federated split learning.\n"
        f"Context: dataset={dataset}; fl_mode={fl_mode}; model={model}; client_profile=low.\n"
        "Return ONLY JSON with fields {client, server, mu}; no extra text."
    )

def valid_hps(hps: Any) -> bool:
    return isinstance(hps, dict) and all(k in hps for k in REQ_KEYS)

def generate_examples(
    data: Dict[str, Any],
    mode: str = "latest_only",
    min_test_acc: float = None
) -> List[Dict[str, Any]]:
    """
    Build chat-style examples; later we also render instruction-style from them.
    Each example: {"messages":[{"role":"user","content":ctx}], "assistant": "<json>", "metadata": {...}}
    """
    meta = data.get("meta", {})
    examples: List[Dict[str, Any]] = []

    for epoch in data.get("epochs", []):
        epoch_id = epoch.get("global_epoch")
        for cluster in epoch.get("clusters", []):
            cluster_id = cluster.get("cluster_id")
            for client in cluster.get("clients", []):
                ctx = client.get("context") or build_fallback_context(meta)

                # >>> NEW: read both accuracies and drop if both are None
                train_acc = client.get("train_acc")
                test_acc  = client.get("test_acc")
                if train_acc is None and test_acc is None:
                    continue
                # <<<

                # keep your optional min_test_acc filter
                if min_test_acc is not None and isinstance(test_acc, (int, float)):
                    if test_acc < min_test_acc:
                        continue

                # choose targets depending on mode
                if mode == "per_event":
                    hp_list = client.get("hp_events") or []
                    if not hp_list and client.get("hps"):
                        hp_list = [client["hps"]]
                else:  # latest_only
                    hp = client.get("hps")
                    hp_list = [hp] if hp else []

                num_events = len(hp_list)
                analyzer_events = client.get("analyzer_events") or []
                analyzer_count = len(analyzer_events)

                for eidx, hps in enumerate(hp_list):
                    if not valid_hps(hps):
                        continue

                    record_meta = {
                        "model": meta.get("model"),
                        "dataset": meta.get("dataset"),
                        "imbalance_ratio": meta.get("imbalance_ratio"),
                        "fl_mode": meta.get("fl_mode"),
                        "total_epochs": meta.get("total_epochs"),
                        "hpo_strategy": meta.get("hpo_strategy"),
                        "epoch": epoch_id,
                        "cluster_id": cluster_id,
                        "client_id": client.get("client_id"),
                        # >>> NEW: use locals so they're always present
                        "train_acc": train_acc,
                        "test_acc":  test_acc,
                        # <<<
                        "event_idx": eidx,
                        "num_events": num_events,
                        "analyzer_events_count": analyzer_count,
                    }
                    if isinstance(client.get("reasoning"), str) and client["reasoning"].strip():
                        record_meta["reasoning"] = client["reasoning"].strip()

                    examples.append({
                        "messages": [{"role": "user", "content": ctx}],
                        "assistant": to_minified_json(hps),
                        "metadata": record_meta
                    })

    return examples


def write_chat_jsonl(examples: List[Dict[str, Any]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return len(examples)

def write_instruct_jsonl(examples: List[Dict[str, Any]], out_path: Path) -> int:
    instr_examples = []
    for ex in examples:
        ctx = ex["messages"][0]["content"]
        instr, inp = split_instruction_input(ctx)
        instr_examples.append({
            "instruction": instr or "Suggest hyperparameters for the next local training round in federated split learning.",
            "input": inp,
            "output": ex["assistant"],  # strictly the JSON string
            "metadata": ex["metadata"]
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in instr_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return len(instr_examples)

def main():
    ap = argparse.ArgumentParser(description="Generate SFT datasets (chat + instruction) from parsed FL HPO JSON.")
    ap.add_argument("-i", "--input",  default="/u/aalasif/SLM_FL_HPO/code/parse/epoch_client_hps.json",
                    help="Path to parsed JSON (epochs/clusters/clients).")
    ap.add_argument("--out_chat", default="/u/aalasif/SLM_FL_HPO/code/parse/sft_chat.jsonl",
                    help="Output path for chat-style JSONL.")
    ap.add_argument("--out_instruct", default="/u/aalasif/SLM_FL_HPO/code/parse/sft_instruct.jsonl",
                    help="Output path for instruction-style JSONL.")
    ap.add_argument("--mode", choices=["latest_only", "per_event"], default="latest_only",
                    help="Emit only the latest hps for each client (default) or one row per hp event.")
    ap.add_argument("--min_test_acc", type=float, default=None,
                    help="If set, drop examples with test_acc < this value.")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    examples = generate_examples(data, mode=args.mode, min_test_acc=args.min_test_acc)

    n_chat = write_chat_jsonl(examples, Path(args.out_chat))
    n_instr = write_instruct_jsonl(examples, Path(args.out_instruct))
    print(f"Wrote {n_chat} chat examples to {args.out_chat}")
    print(f"Wrote {n_instr} instruction examples to {args.out_instruct}")

if __name__ == "__main__":
    main()
