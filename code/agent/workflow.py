# # # code/agent/workflow.py

import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, Dict, List

from .hp_agent import HPAgent
from .analyzer_agent import AnalyzerAgent
from ssfl.trainer_utils import log_epoch_metrics 
import time
from . import shared_state
from .policy_adapter import should_update_hps_for_client, mark_hpo_updated






# --- Final-round detector (robust to missing keys) ---
def _is_final_round(state: dict) -> bool:
    """
    Return True iff this item corresponds to the final training round.
    Prefers ('round_idx','total_rounds'); falls back to ('global_epoch','global_epochs').
    """
    r = state.get("round_idx")
    T = state.get("total_rounds")
    if r is not None and T is not None:
        try:
            return int(r) >= int(T)
        except Exception:
            pass

    # Fallback: global_epoch is 0-based; global_epochs is the total count
    ge = state.get("global_epoch")
    gE = state.get("global_epochs") or state.get("total_rounds")
    if ge is not None and gE is not None:
        try:
            return int(ge) + 1 >= int(gE)
        except Exception:
            pass

    # If we cannot determine, assume not final (safe default)
    return False


# --- On-the-fly adaptation helpers (reward & stability) ---
def _instability_index(state: dict, W: int = 5) -> float:
    losses = (state.get("recent_losses") or [])[-W:]
    osc = 0.0
    if len(losses) >= 3:
        m = sum(losses) / len(losses)
        var = sum((x - m) ** 2 for x in losses) / len(losses)
        std = var ** 0.5
        # normalize oscillation by (|mean|+1e-6) to be scale-aware
        osc = std / (abs(m) + 1e-6)

    prev_hps = state.get("prev_hps") or {}
    cur_hps  = state.get("hps_used") or {}

    lr_t  = (cur_hps.get("client") or {}).get("learning_rate")
    lr_tm = (prev_hps.get("client") or {}).get("learning_rate")
    b_t   = (cur_hps.get("client") or {}).get("batch_size")
    b_tm  = (prev_hps.get("client") or {}).get("batch_size")

    import math
    hp_jump = 0.0
    if lr_t and lr_tm and lr_t > 0 and lr_tm > 0:
        hp_jump += abs(math.log(lr_t) - math.log(lr_tm))
    if b_t and b_tm and b_tm > 0:
        hp_jump += abs(b_t - b_tm) / b_tm

    mu_used = float(state.get("mu_used") or 0.0)

    # squash to [0,1]-ish
    osc_norm = min(1.0, osc / 0.2)        # std equal to 20% of mean is “large”
    hp_norm  = min(1.0, hp_jump / 1.0)
    mu_norm  = min(1.0, mu_used / 0.1)

    return (osc_norm + hp_norm + mu_norm) / 3.0



def _lyapunov_pass(state: dict, beta: float = 0.3, base_eps: float = 5e-3, W: int = 5) -> bool:
    """
    V = EMA(loss) + β·oscillation; accept if ΔV <= ε_rel.
    ε_rel adapts to noise level (recent std) so early noise doesn't block.
    Also: warmup guard + fast-pass if accuracy improved.
    """
    losses = (state.get("recent_losses") or [])[-W:]
    accs   = (state.get("recent_accs") or [])[-2:]

    # --- Warmup: don't block with tiny history ---
    if len(losses) < 3:
        return True

    # --- Fast pass if accuracy improved ---
    if len(accs) == 2 and (accs[-1] - accs[-2]) > 0:
        return True

    # EMA
    alpha = 0.4  # smoother than 0.6
    ema = 0.0
    for x in losses:
        ema = alpha * x + (1 - alpha) * ema

    # oscillation & std
    m   = sum(losses) / len(losses)
    var = sum((x - m) ** 2 for x in losses) / len(losses)
    std = var ** 0.5

    Vt = ema + beta * var

    # crude one-step lookahead = current trend
    d_last = losses[-1] - losses[-2]
    ema_next = alpha * (losses[-1] + d_last) + (1 - alpha) * ema
    Vnext = ema_next + beta * var  # keep same var for a 1-step lookahead

    dV = Vnext - Vt

    # --- Relative tolerance: scale epsilon by noise level ---
    eps = base_eps + 0.25 * std   # tolerate more when stream is noisy

    # Optional: print debug once in a while
    print(f"[Lyapunov] Vt={Vt:.4f} Vnext={Vnext:.4f} dV={dV:.4f} std={std:.4f} eps={eps:.4f}")

    return dV <= eps


def _lyapunov_pass_more_strict(state: dict, beta: float = 1.0, eps: float = 1e-3, W: int = 5) -> bool:
    """V = EMA(loss) + β·oscillation; accept if ΔV <= ε (simple local lookahead)."""
    losses = (state.get("recent_losses") or [])[-W:]
    if not losses:
        return True

    # EMA
    ema = 0.0
    alpha = 0.6
    for x in losses:
        ema = alpha * x + (1 - alpha) * ema

    # oscillation
    m = sum(losses) / len(losses)
    osc = sum((x - m) ** 2 for x in losses) / len(losses)

    Vt = ema + beta * osc
    if len(losses) >= 2:
        d_last = losses[-1] - losses[-2]
        ema_next = alpha * (losses[-1] + d_last) + (1 - alpha) * ema
    else:
        ema_next = ema
    Vnext = ema_next + beta * osc

    return (Vnext - Vt) <= eps


# --- HPOState is now correct ---
# It has a dedicated 'hps' field for the output of the suggest_node.
class HPOState(TypedDict, total=False):
    client_id: int
    cluster_id: int
    model_name: str
    dataset_name: str
    peer_history: List[dict]
    global_epoch: int
    hpo_report: dict
    search_space: dict
    results: Dict[str, Any]
    current_hps: Dict[str, Any]
    last_analysis: Dict[str, Any]
    hps: Dict[str, Any] # <-- Field for the final chosen HPs

    training_time: float

    # These keys will be added by our nodes
    llm_analysis_latency: float
    llm_suggestion_latency: float

    analysis_prompt_tokens: int
    analysis_completion_tokens: int
    suggestion_prompt_tokens: int
    suggestion_completion_tokens: int

    detailed_log_filename: str 


hp_agent = HPAgent()
analyzer_agent = AnalyzerAgent()

def analyze_node(state: HPOState) -> HPOState:
    print(f">>> Graph Node: ANALYZE for Client {state['client_id']}")

    if _is_final_round(state) or state.get("phase") == "finalize":
        print(f"[TRACE][GRAPH][finalize] ANALYZE skipped (final round) for client {state['client_id']}")
        return state
    

    start_time = time.time()
    new_search_space, reasoning, usage = analyzer_agent.analyze(
        client_id=state['client_id'],
        cluster_id=state['cluster_id'],
        model_name=state['model_name'],
        dataset_name=state['dataset_name'],
        results=state['results'],
        current_hps=state['current_hps'],
        search_space=state['search_space'],
        global_epoch=state['global_epoch'],
        local_epochs=state['current_hps'].get('client',{}).get('local_epochs', 1)
    )
    end_time = time.time()
    #print(f"  ... LLM response received. Analysis Latency: {end_time - start_time:.2f} seconds.")
    latency = end_time - start_time
    print(f"  ... LLM response received. Analysis Latency: {latency:.2f} seconds.")
    
    state['search_space'] = new_search_space
    state['last_analysis'] = reasoning

    # --- KEY CHANGE 1: Store the analysis latency in the state package ---
    state['llm_analysis_latency'] = latency
    # state['training_time'] = state.get('training_time', 0.0)
    # state['accuracy'] = state.get('results', {}).get('test_acc', [0.0])[-1]
    state['analysis_prompt_tokens'] = usage.get('prompt_tokens', 0)
    state['analysis_completion_tokens'] = usage.get('completion_tokens', 0)

    return state

def suggest_node(state: HPOState) -> HPOState:
    print(f"\n>>> Graph Node: SUGGEST for Client {state['client_id']}")

    # Final round / finalize phase: never call LLM
    if _is_final_round(state) or state.get("phase") == "finalize":
        print(f"[TRACE][GRAPH][finalize] SUGGEST skipped (final round) for client {state['client_id']}")
        if "hps" not in state or not state["hps"]:
            state["hps"] = state.get("current_hps", state.get("hps", {}))
        state['llm_suggestion_latency'] = 0.0
        return state

    # Guard duplicate post_analyze suggest within the same (client,epoch)
    if not shared_state.mark_suggest_once(state['client_id'], state['global_epoch'], "post_analyze"):
        print(f"[SKIP] Duplicate post_analyze SUGGEST for client {state['client_id']} epoch {state['global_epoch']}")
        return state

    # Pull freshest per-client metrics published by strategies
    try:
        from agent import shared_state as _ss
        cm = getattr(_ss, "CLIENT_METRICS", {}).get(state["client_id"])
        if cm:
            state.setdefault("recent_accs", cm.get("recent_accs", state.get("recent_accs", [])))
            state.setdefault("recent_losses", cm.get("recent_losses", state.get("recent_losses", [])))
            state.setdefault("hps_used", cm.get("hps_used", state.get("hps_used", {})))
            state.setdefault("prev_hps", cm.get("prev_hps", state.get("prev_hps", {})))
            state.setdefault("mu_used", cm.get("mu_used", state.get("mu_used", 0.0)))
    except Exception:
        pass

    # === Compute reward & stability (unchanged reward; Lyapunov now uses YAML) ===
    acc_hist = (state.get("recent_accs") or [])[-2:]
    delta_acc = (acc_hist[-1] - acc_hist[-2]) if len(acc_hist) == 2 else 0.0

    reward_cfg = shared_state.CONFIG.get("stability", {}).get("reward", {})
    lam = float(reward_cfg.get("lambda_penalty", 0.3))

    #lam = float(state.get("lambda_penalty", 0.3))
    instability = _instability_index(state)
    reward = float(delta_acc - lam * instability)
    # optional scaling knob (kept separate from formula)
    reward *= float(reward_cfg.get("scale", 1.0))
    state["reward"] = reward



    # --- NEW: read stability config from YAML and run Lyapunov with those params
    stab = shared_state.CONFIG.get("stability", {})
    pre  = stab.get("pre_gate", {})
    lya  = stab.get("lyapunov", {})

    state["lyapunov_pass"] = _lyapunov_pass(
        state,
        beta=float(lya.get("beta", 0.3)),
        base_eps=float(lya.get("base_eps", 5e-3)),
        W=int(lya.get("window", 5)),
    )

    # Feedback to mailbox (consumed in HPAgent.suggest for tiny LoRA update)
    shared_state.attach_feedback(
        state['client_id'],
        reward=state['reward'],
        lyapunov_pass=state['lyapunov_pass'],
    )

    # --- NEW: PRE-GATE BEFORE ANY LLM CALL ---
    if not should_update_hps_for_client(
        client_id=state['client_id'],
        round_idx=int(state['global_epoch']),
        delta_acc=float(delta_acc),
        lyapunov_pass=bool(state["lyapunov_pass"]),
        min_round_gap=int(pre.get("min_round_gap", 1)),
        min_delta=float(pre.get("min_delta", 0.0)),
        require_lyapunov=bool(pre.get("require_lyapunov", False)),
    ):
        print(f"[SUGGEST] skip LLM: pre-gate false (cid={state['client_id']}, epoch={state['global_epoch']})")
        # keep HPs unchanged for next epoch
        if "hps" not in state or not state["hps"]:
            state["hps"] = state.get("current_hps", state.get("hps", {}))
        next_epoch = state['global_epoch'] + 1
        shared_state.set_next_hps(state['client_id'], next_epoch, state["hps"])
        state['llm_suggestion_latency'] = 0.0
        return state

    # === Passed pre-gate → single-shot HP suggest ===
    start_time = time.time()
    hps, usage = hp_agent.suggest(
        client_id=state['client_id'],
        cluster_id=state['cluster_id'],
        model_name=state['model_name'],
        dataset_name=state['dataset_name'],
        hpo_report=state['hpo_report'],
        search_space=state['search_space'],
        analysis_from_last_round=state.get('last_analysis'),
        peer_history=state.get('peer_history')
    )
    end_time = time.time()
    suggestion_latency = end_time - start_time
    print(f"  ... LLM response received. HP Suggestion Latency: {suggestion_latency:.2f} seconds.")

    # Save HPs and publish for next epoch
    state['hps'] = hps
    next_epoch = state['global_epoch'] + 1
    shared_state.set_next_hps(state['client_id'], next_epoch, hps)

    state['llm_suggestion_latency'] = suggestion_latency
    state["last_prompt"] = getattr(hp_agent, "last_prompt", state.get("last_prompt", ""))
    state["last_response"] = getattr(hp_agent, "last_response", state.get("last_response", ""))
    state['suggestion_prompt_tokens'] = usage.get('prompt_tokens', 0)
    state['suggestion_completion_tokens'] = usage.get('completion_tokens', 0)

    # --- NEW: remember that we actually updated HPs this round ---
    mark_hpo_updated(state['client_id'], int(state['global_epoch']))

    return state


def suggest_node_old(state: HPOState) -> HPOState:
    print(f"\n>>> Graph Node: SUGGEST for Client {state['client_id']}")


    if _is_final_round(state) or state.get("phase") == "finalize":
        print(f"[TRACE][GRAPH][finalize] SUGGEST skipped (final round) for client {state['client_id']}")
        if "hps" not in state or not state["hps"]:
            state["hps"] = state.get("current_hps", state.get("hps", {}))
        state['llm_suggestion_latency'] = 0.0
        return state

    # 1) GUARD FIRST — skip duplicates before spending an LLM call
    if not shared_state.mark_suggest_once(state['client_id'], state['global_epoch'], "post_analyze"):
        print(f"[SKIP] Duplicate post_analyze SUGGEST for client {state['client_id']} epoch {state['global_epoch']}")
        return state

    # Pull freshest per-client metrics from shared_state (published by strategies)
    try:
        from agent import shared_state as _ss
        cm = getattr(_ss, "CLIENT_METRICS", {}).get(state["client_id"])
        if cm:
            state.setdefault("recent_accs", cm.get("recent_accs", state.get("recent_accs", [])))
            state.setdefault("recent_losses", cm.get("recent_losses", state.get("recent_losses", [])))
            state.setdefault("hps_used", cm.get("hps_used", state.get("hps_used", {})))
            state.setdefault("prev_hps", cm.get("prev_hps", state.get("prev_hps", {})))
            state.setdefault("mu_used", cm.get("mu_used", state.get("mu_used", 0.0)))
    except Exception:
        pass

    # === Compute reward & stability flag for on-the-fly adaptation ===
    acc_hist = (state.get("recent_accs") or [])[-2:]
    delta_acc = (acc_hist[-1] - acc_hist[-2]) if len(acc_hist) == 2 else 0.0

    




    lam = float(state.get("lambda_penalty", 0.3))  # optional: from config/state
    instability = _instability_index(state)
    reward = float(delta_acc - lam * instability)
    state["reward"] = reward

    # Lyapunov gate
    state["lyapunov_pass"] = _lyapunov_pass(state)

    # New: pass feedback to the shared mailbox (consumed in HPAgent.suggest)
    shared_state.attach_feedback(
        state['client_id'],
        reward=state['reward'],
        lyapunov_pass=state['lyapunov_pass'],
    )




    start_time = time.time()
    # --- THIS IS THE FIX ---
    # The 'suggest' function now gets the refined search_space from the 'analyze' node.
    # Its output is placed into the 'hps' key, NOT 'search_space'.
    hps, usage = hp_agent.suggest(
        client_id=state['client_id'],
        cluster_id=state['cluster_id'],
        model_name=state['model_name'],
        dataset_name=state['dataset_name'],
        hpo_report=state['hpo_report'],
        search_space=state['search_space'],
        analysis_from_last_round=state.get('last_analysis'),
        peer_history=state.get('peer_history')
    )
    end_time = time.time()
    suggestion_latency = end_time - start_time
    print(f"  ... LLM response received. HP Suggestion Latency: {suggestion_latency:.2f} seconds.")

    # --- THIS IS THE CRITICAL DEBUGGING STEP ---
    # Add this line to see what 'usage' contains inside the workflow node.
    #print(f"  [WORKFLOW DEBUG] Usage data received in suggest_node: {usage}")
    # ---------------------------------------------

   
    #print(f"  ... LLM response received. HP Suggestion Latency: {end_time - start_time:.2f} seconds.")
    state['hps'] = hps # <-- Put the result in the 'hps' key.

    # Make these HPs available to the strategy for the NEXT epoch
    next_epoch = state['global_epoch'] + 1
    shared_state.set_next_hps(state['client_id'], next_epoch, hps)


    state['llm_suggestion_latency'] = suggestion_latency

    state["last_prompt"] = getattr(hp_agent, "last_prompt", state.get("last_prompt", ""))
    state["last_response"] = getattr(hp_agent, "last_response", state.get("last_response", ""))

    state['suggestion_prompt_tokens'] = usage.get('prompt_tokens', 0)
    state['suggestion_completion_tokens'] = usage.get('completion_tokens', 0)

    return state

# --- NEW NODE ---
def log_metrics_node(state: HPOState) -> HPOState:
    """
    Final node that receives the fully populated state and logs the metrics.
    """
    print(f"[CPU Worker]: Logging metrics for Client {state['client_id']}...")

    results = state.get('results', {})
    
    # Extract accuracy from the results dictionary
    accuracy = results.get('test_acc', [0.0])[-1]
    train_loss = results.get('train_loss', [0.0])[-1]
    test_loss = results.get('test_loss', [0.0])[-1]

    hps = state.get('hps', {})
    client_hps = hps.get('client', {})
    
    metrics_to_log = {
        'training_time': round(state.get('training_time', 0.0), 4),
        'analysis_latency': round(state.get('llm_analysis_latency', 0.0), 4),
        'suggestion_latency': round(state.get('llm_suggestion_latency', 0.0), 4),
        'accuracy': round(accuracy, 4),
        'train_loss': round(train_loss, 4),      # <-- This adds the train loss
        'test_loss': round(test_loss, 4),  
        'lr': client_hps.get('learning_rate'),
        'batch_size': client_hps.get('batch_size'),
        'local_epochs': client_hps.get('local_epochs'),
        

        # Add the token counts from the state to the log
        'analysis_prompt_tokens': state.get('analysis_prompt_tokens', 0),
        'analysis_completion_tokens': state.get('analysis_completion_tokens', 0),
        'suggestion_prompt_tokens': state.get('suggestion_prompt_tokens', 0),
        'suggestion_completion_tokens': state.get('suggestion_completion_tokens', 0),

        'cluster_id': state.get('cluster_id', -1),
    }
    
    log_epoch_metrics(
        detailed_log_filename=state['detailed_log_filename'],
        model_name=state['model_name'],
        dataset_name=state['dataset_name'],
        epoch=state['global_epoch'],
        client_id=state['client_id'],
        metrics=metrics_to_log
    )
    
    return state

def create_cpu_graph():
    workflow = StateGraph(HPOState)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("suggest", suggest_node)
    workflow.add_node("log_metrics", log_metrics_node) # <-- Add the new node


    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "suggest")
    workflow.add_edge("suggest", "log_metrics") # <-- Connect suggest to log
    workflow.add_edge("log_metrics", END)      # <-- End after logging
    #workflow.add_edge("suggest", END)
    return workflow.compile()