#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple

REQ_KEYS = ("client", "server", "mu")

def to_minified_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

def read_parsed(in_path: Path) -> Dict[str, Any]:
    with in_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _get_hps(client: Dict[str, Any]) -> Dict[str, Any]:
    """Support both new 'hps' and older 'chosen_hps' field names."""
    hps = client.get("hps")
    if not hps:
        hps = client.get("chosen_hps")
    return hps or {}

def collect_candidates(data: Dict[str, Any], debug: bool=False) -> List[Dict[str, Any]]:
    """Flatten epochs→clusters→clients into a list of candidate records."""
    meta = data.get("meta", {})
    rows = []

    stats = {
        "total_clients_seen": 0,
        "kept": 0,
        "skip_no_hps": 0,
        "skip_bad_hps": 0,
        "skip_no_test_acc": 0,
        "skip_bad_client_id": 0,
        "skip_other": 0,
    }

    for epoch in data.get("epochs", []):
        epoch_id = epoch.get("global_epoch")
        for cluster in epoch.get("clusters", []):
            cluster_id = cluster.get("cluster_id")
            for client in cluster.get("clients", []):
                stats["total_clients_seen"] += 1

                hps = _get_hps(client)
                if not hps:
                    stats["skip_no_hps"] += 1
                    if debug:
                        print(f"[SKIP no_hps] epoch={epoch_id} cluster={cluster_id} client={client.get('client_id')}")
                    continue

                if not isinstance(hps, dict) or not all(k in hps for k in REQ_KEYS):
                    stats["skip_bad_hps"] += 1
                    if debug:
                        miss = [k for k in REQ_KEYS if k not in hps] if isinstance(hps, dict) else "not_dict"
                        print(f"[SKIP bad_hps:{miss}] epoch={epoch_id} cluster={cluster_id} client={client.get('client_id')}")
                    continue

                test_acc = client.get("test_acc", None)
                if test_acc is None:
                    stats["skip_no_test_acc"] += 1
                    if debug:
                        print(f"[SKIP no_test_acc] epoch={epoch_id} cluster={cluster_id} client={client.get('client_id')}")
                    continue

                # Normalize client_id for grouping
                cid_raw = client.get("client_id")
                try:
                    cid = int(cid_raw)
                except Exception:
                    stats["skip_bad_client_id"] += 1
                    if debug:
                        print(f"[SKIP bad_client_id] epoch={epoch_id} cluster={cluster_id} client_id={cid_raw}")
                    continue

                try:
                    rows.append({
                        "epoch": epoch_id,
                        "cluster_id": cluster_id,
                        "client_id": cid,
                        "context": client.get("context", ""),
                        "hps": hps,
                        "test_acc": float(test_acc),
                        "train_acc": client.get("train_acc", None),
                        "reasoning": client.get("reasoning", None),
                        "meta": meta,
                    })
                    stats["kept"] += 1
                except Exception as e:
                    stats["skip_other"] += 1
                    if debug:
                        print(f"[SKIP other:{e}] epoch={epoch_id} cluster={cluster_id} client={cid_raw}")

    if debug:
        print("\n[DIAG] collect_candidates summary:")
        print(json.dumps(stats, indent=2))
        counts = defaultdict(int)
        for r in rows:
            counts[r["client_id"]] += 1
        print("[DIAG] per-client counts:", dict(counts))
    return rows

def group_by_client(rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["client_id"]].append(r)
    return groups

def build_pairs_for_client(
    items: List[Dict[str, Any]],
    min_gap: float = 0.0,
    pair_mode: str = "best_vs_rest",
    debug: bool=False
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Returns list of (chosen, rejected) pairs for a single client's items.
      - "best_vs_rest": pick highest test_acc as chosen; pair vs every lesser item
      - "all_pairs": all pairwise combinations, higher acc is chosen
    """
    if len(items) < 2:
        return []

    items_sorted = sorted(items, key=lambda x: x["test_acc"], reverse=True)
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    if pair_mode == "best_vs_rest":
        best = items_sorted[0]
        for other in items_sorted[1:]:
            gap = best["test_acc"] - other["test_acc"]
            if gap >= min_gap:
                pairs.append((best, other))
            elif debug:
                print(f"[SKIP pair gap] client={best['client_id']} best={best['test_acc']} other={other['test_acc']} gap={gap} < min_gap={min_gap}")
    else:
        n = len(items_sorted)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = items_sorted[i], items_sorted[j]
                if a["test_acc"] == b["test_acc"]:
                    if debug:
                        print(f"[SKIP equal_acc] client={a['client_id']} acc={a['test_acc']}")
                    continue
                chosen, rejected = (a, b) if a["test_acc"] > b["test_acc"] else (b, a)
                gap = chosen["test_acc"] - rejected["test_acc"]
                if gap >= min_gap:
                    pairs.append((chosen, rejected))
                elif debug:
                    print(f"[SKIP pair gap] client={a['client_id']} gap={gap} < min_gap={min_gap}")

    if debug:
        print(f"[DIAG] client={items_sorted[0]['client_id']} produced {len(pairs)} pairs (mode={pair_mode})")
    return pairs

def write_dpo_flat(
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    out_path: Path
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for chosen, rejected in pairs:
            rec = {
                "prompt": chosen.get("context", ""),
                "chosen": to_minified_json(chosen["hps"]),
                "rejected": to_minified_json(rejected["hps"]),
                "metadata": {
                    "dataset": chosen["meta"].get("dataset"),
                    "model": chosen["meta"].get("model"),
                    "fl_mode": chosen["meta"].get("fl_mode"),
                    "imbalance_ratio": chosen["meta"].get("imbalance_ratio"),
                    "hpo_strategy": chosen["meta"].get("hpo_strategy"),
                    "client_id": chosen.get("client_id"),
                    "chosen_epoch": chosen.get("epoch"),
                    "rejected_epoch": rejected.get("epoch"),
                    "chosen_test_acc": chosen.get("test_acc"),
                    "rejected_test_acc": rejected.get("test_acc"),
                    "cluster_id_chosen": chosen.get("cluster_id"),
                    "cluster_id_rejected": rejected.get("cluster_id"),
                    "reasoning_chosen": chosen.get("reasoning"),
                    "reasoning_rejected": rejected.get("reasoning"),
                }
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n

def write_dpo_chat(
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    out_path: Path
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for chosen, rejected in pairs:
            rec = {
                "messages": [{"role": "user", "content": chosen.get("context", "")}],
                "chosen": to_minified_json(chosen["hps"]),
                "rejected": to_minified_json(rejected["hps"]),
                "metadata": {
                    "dataset": chosen["meta"].get("dataset"),
                    "model": chosen["meta"].get("model"),
                    "fl_mode": chosen["meta"].get("fl_mode"),
                    "imbalance_ratio": chosen["meta"].get("imbalance_ratio"),
                    "hpo_strategy": chosen["meta"].get("hpo_strategy"),
                    "client_id": chosen.get("client_id"),
                    "chosen_epoch": chosen.get("epoch"),
                    "rejected_epoch": rejected.get("epoch"),
                    "chosen_test_acc": chosen.get("test_acc"),
                    "rejected_test_acc": rejected.get("test_acc"),
                    "cluster_id_chosen": chosen.get("cluster_id"),
                    "cluster_id_rejected": rejected.get("cluster_id"),
                    "reasoning_chosen": chosen.get("reasoning"),
                    "reasoning_rejected": rejected.get("reasoning"),
                }
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser(description="Build DPO-style dataset from parsed FL HPO JSON (with fallbacks & debug).")
    ap.add_argument("-i", "--input", default="/u/aalasif/SLM_FL_HPO/code/parse/epoch_client_hps.json",
                    help="Path to parsed JSON (epochs/clusters/clients).")
    ap.add_argument("--out_flat", default="/u/aalasif/SLM_FL_HPO/code/parse/dpo_flat.jsonl",
                    help="Output path for flat DPO JSONL.")
    ap.add_argument("--out_chat", default="/u/aalasif/SLM_FL_HPO/code/parse/dpo_chat.jsonl",
                    help="Output path for chat DPO JSONL.")
    ap.add_argument("--pair_mode", choices=["best_vs_rest", "all_pairs"], default="best_vs_rest",
                    help="Pairing strategy across a client's epochs.")
    ap.add_argument("--min_gap", type=float, default=0.0,
                    help="Minimum (chosen_test_acc - rejected_test_acc) required to make a pair.")
    ap.add_argument("--debug", action="store_true", help="Print diagnostics for skipped items and grouping.")
    args = ap.parse_args()

    data = read_parsed(Path(args.input))
    rows = collect_candidates(data, debug=args.debug)
    groups = group_by_client(rows)

    all_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for client_id, items in groups.items():
        all_pairs.extend(build_pairs_for_client(
            items, min_gap=args.min_gap, pair_mode=args.pair_mode, debug=args.debug
        ))

    n_flat = write_dpo_flat(all_pairs, Path(args.out_flat))
    n_chat = write_dpo_chat(all_pairs, Path(args.out_chat))

    print(f"Built {len(all_pairs)} total DPO pairs")
    print(f"Wrote {n_flat} pairs to {args.out_flat} (flat)")
    print(f"Wrote {n_chat} pairs to {args.out_chat} (chat)")

if __name__ == "__main__":
    main()
