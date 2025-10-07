# # # code/agent/workflow.py

import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, Any, Dict, List

from .hp_agent import HPAgent
from .analyzer_agent import AnalyzerAgent
from ssfl.trainer_utils import log_epoch_metrics 
import time
from . import shared_state

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

    # 1) GUARD FIRST — skip duplicates before spending an LLM call
    if not shared_state.mark_suggest_once(state['client_id'], state['global_epoch'], "post_analyze"):
        print(f"[SKIP] Duplicate post_analyze SUGGEST for client {state['client_id']} epoch {state['global_epoch']}")
        return state



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