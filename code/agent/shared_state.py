# agent/shared_state.py

import json
import os
import time
import uuid
from queue import Queue

# ========= Existing queue (unchanged) =========
# Holds results from the trainer. Format: (client_id, results_dict)
results_queue = Queue()

# ========= Optional in-memory snapshots (same-process only) =========
HP_AGENT_STATS = {}
ANALYZER_AGENT_STATS = {}

# ========= Paths =========
_STATS_DIR = os.path.dirname(__file__)

# Per-process last-snapshot files (used to compute deltas)
def _last_snap_path(name: str, pid: int) -> str:
    base = "hp" if name.lower() == "hp" else "analyzer"
    return os.path.join(_STATS_DIR, f"_{base}_last_{pid}.json")

# Append-only JSONL event logs (safe for cross-process aggregation)
_HP_LOG = os.path.join(_STATS_DIR, "_hp_agent_events.jsonl")
_ANALYZER_LOG = os.path.join(_STATS_DIR, "_analyzer_agent_events.jsonl")

# ========= Internal helpers =========

def _read_json(path: str) -> dict:
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _write_json(path: str, obj: dict) -> None:
    try:
        with open(path, "w") as f:
            json.dump(obj, f)
    except Exception:
        pass

def _append_jsonl(path: str, obj: dict) -> None:
    try:
        with open(path, "a") as f:
            f.write(json.dumps(obj) + "\n")
    except Exception:
        pass

def _delta_from_snapshots(prev: dict, curr: dict, keys: list[str]) -> dict:
    delta = {}
    for k in keys:
        prev_v = int(prev.get(k, 0))
        curr_v = int(curr.get(k, 0))
        d = curr_v - prev_v
        if d < 0:
            # If counters reset (e.g., process restarted), treat as full snapshot
            d = curr_v
        delta[k] = d
    return delta

# ========= Public API used by agents =========

def save_stats(name: str, snapshot: dict) -> None:
    """
    Called by agents with their current cumulative counters (snapshot).
    We turn that into a per-call DELTA and append it to a process-safe JSONL.

    You don't need to change agent code if they already call save_stats("hp", stats)
    or save_stats("analyzer", stats).
    """
    pid = os.getpid()
    last_path = _last_snap_path(name, pid)
    prev = _read_json(last_path)

    # keys we aggregate
    if name.lower() == "hp":
        keys = [
            "hp_calls",
            "hp_success",
            "hp_json_errors",
            "hp_fallback_used",
            "clamps",
            "invalid_choice_fixed",
            "unknown_param_ignored",
        ]
        log_path = _HP_LOG
    else:
        keys = [
            "analyze_calls",
            "analyze_success",
            "analyze_json_errors",
            "actions_total",
            "actions_applied",
            "actions_skipped_malformed",
            "actions_skipped_invalid_target",
            "actions_skipped_invalid_key",
            "actions_values_list_type_error",
            "actions_numeric_values_coerced",
        ]
        log_path = _ANALYZER_LOG

    # compute delta (handles counter resets)
    delta = _delta_from_snapshots(prev, snapshot, keys)

    # append event only if there is any nonzero delta
    if any(int(delta[k]) != 0 for k in keys):
        evt = {
            **{k: int(delta[k]) for k in keys},
            "_ts": time.time(),
            "_pid": pid,
            "_id": str(uuid.uuid4()),
        }
        _append_jsonl(log_path, evt)

    # update last snapshot
    _write_json(last_path, {k: int(snapshot.get(k, 0)) for k in keys})

def aggregate_hp_events() -> dict:
    """Read all HP JSONL events and sum them."""
    totals = {
        "hp_calls": 0,
        "hp_success": 0,
        "hp_json_errors": 0,
        "hp_fallback_used": 0,
        "clamps": 0,
        "invalid_choice_fixed": 0,
        "unknown_param_ignored": 0,
    }
    try:
        if os.path.exists(_HP_LOG):
            with open(_HP_LOG, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    evt = json.loads(line)
                    for k in totals:
                        if k in evt:
                            totals[k] += int(evt[k])
    except Exception:
        pass
    return totals

# Add to agent/shared_state.py
def reset_aggregates() -> None:
    """Clear aggregated JSONL event logs for a fresh run."""
    for path in (_HP_LOG, _ANALYZER_LOG):
        try:
            # Truncate rather than remove, to keep file permissions/paths stable
            with open(path, "w") as f:
                f.write("")
        except Exception:
            pass

def aggregate_analyzer_events() -> dict:
    """Read all Analyzer JSONL events and sum them."""
    totals = {
        "analyze_calls": 0,
        "analyze_success": 0,
        "analyze_json_errors": 0,
        "actions_total": 0,
        "actions_applied": 0,
        "actions_skipped_malformed": 0,
        "actions_skipped_invalid_target": 0,
        "actions_skipped_invalid_key": 0,
        "actions_values_list_type_error": 0,
        "actions_numeric_values_coerced": 0,
    }
    try:
        if os.path.exists(_ANALYZER_LOG):
            with open(_ANALYZER_LOG, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    evt = json.loads(line)
                    for k in totals:
                        if k in evt:
                            totals[k] += int(evt[k])
    except Exception:
        pass
    return totals

# --- Next-epoch HP mailbox ---
_NEXT_HPS = {}  # key: (client_id, epoch) -> hps dict

def set_next_hps(client_id: int, epoch: int, hps: dict) -> None:
    _NEXT_HPS[(int(client_id), int(epoch))] = hps

def get_and_pop_next_hps(client_id: int, epoch: int) -> dict | None:
    key = (int(client_id), int(epoch))
    return _NEXT_HPS.pop(key, None)

# --- SUGGEST-once guard (same process) ---
_SUGGEST_MARKS = set()  # tuples of (client_id, epoch, phase)

def mark_suggest_once(client_id: int, epoch: int, phase: str) -> bool:
    key = (int(client_id), int(epoch), str(phase))
    if key in _SUGGEST_MARKS:
        return False
    _SUGGEST_MARKS.add(key)
    return True




# # agent/shared_state.py

# import json
# import os
# from queue import Queue

# # This queue will hold the results from the trainer.
# # The format for items will be: (client_id, results_dict)
# results_queue = Queue()

# # In-memory snapshots (still available if same process imports this)
# HP_AGENT_STATS = {}
# ANALYZER_AGENT_STATS = {}

# # ---- file-based persistence so other processes can read at the end ----
# _STATS_DIR = os.path.dirname(__file__)             # save alongside this file
# _HP_PATH = os.path.join(_STATS_DIR, "_hp_agent_stats.json")
# _ANALYZER_PATH = os.path.join(_STATS_DIR, "_analyzer_agent_stats.json")

# def save_stats(name: str, data: dict) -> None:
#     """Persist stats to a small JSON file so other processes can read them."""
#     path = _HP_PATH if name.lower() == "hp" else _ANALYZER_PATH
#     try:
#         with open(path, "w") as f:
#             json.dump(data, f)
#     except Exception:
#         # Never crash training because stats could not be written
#         pass

# def load_stats(name: str) -> dict:
#     """Load the latest persisted stats; return {} if none exist."""
#     path = _HP_PATH if name.lower() == "hp" else _ANALYZER_PATH
#     try:
#         if os.path.exists(path):
#             with open(path, "r") as f:
#                 return json.load(f)
#     except Exception:
#         pass
#     return {}
