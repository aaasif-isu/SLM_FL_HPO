# code/agent/prompts.py


import json

# --- NEW: Helper function to provide context about cluster capacity ---
def _get_cluster_capacity_string(cluster_id: int) -> str:
    """Provides a human-readable description of the cluster's capacity."""
    if cluster_id == 0:
        return f"Cluster {cluster_id} (Low-Resource): These are the weakest clients. They may require smaller batch sizes or lower learning rates to train stably."
    elif cluster_id == 1:
        return f"Cluster {cluster_id} (Medium-Resource): These clients have moderate capacity."
    else:
        return f"Cluster {cluster_id} (High-Resource): These are the most powerful clients. They can likely handle larger batch sizes and more aggressive learning rates."


def _build_dynamic_search_space_description(search_space: dict) -> str:
    """Helper function to dynamically create a string describing the search space."""
    lines = []
    for top_level_key, hp_group in search_space.items():
        lines.append(f"\n**{top_level_key.replace('_', ' ').title()} Search Space:**")
        if isinstance(hp_group, dict):
            for name, config in hp_group.items():
                if isinstance(config, dict):
                    clean_name = name.replace('_', ' ').title()
                    param_type = config.get('type', 'unknown')
                    if param_type in ['float', 'int']:
                        line = f"- **{clean_name}** (`{param_type}`): from `{config.get('min')}` to `{config.get('max')}`"
                    elif param_type == 'choice':
                        line = f"- **{clean_name}** (`{param_type}`): from options `{config.get('values')}`"
                    else:
                        continue # Skip non-dict items like 'mu' which is handled separately
                    lines.append(line)
    # Handle mu separately if it exists at the top level
    if 'mu' in search_space and isinstance(search_space['mu'], dict):
         lines.append(f"\n**Global Hyperparameters:**")
         mu_config = search_space['mu']
         lines.append(f"- **Mu** (`float`): from `{mu_config.get('min')}` to `{mu_config.get('max')}`")

    return "\n".join(lines)

def get_hp_suggestion_prompt(
    client_id: int, cluster_id: int, model_name: str, dataset_name: str,
    hpo_report: dict, search_space: dict, analysis_from_last_round: dict | None = None,
    peer_history: list | None = None, arc_cfg: int = 0, total_layers: int = 0
) -> str:

    # Build history string (existing logic)
    history_lines = []
    if hpo_report:
        for epoch, report in sorted(hpo_report.items()):
            hps_str = ", ".join(f"{k}={v}" for k, v in report.get('hps_suggested', {}).items())
            acc = report.get('final_test_accuracy', 0.0)
            history_lines.append(f"- In Epoch {epoch + 1}: Used HPs ({hps_str}) and achieved Test Accuracy = {acc:.2f}%")
    else:
        history_lines.append("This is the first round for this client.")
    history_str = "\n".join(history_lines)

    analysis_str = "No analysis from the previous round is available."
    if analysis_from_last_round:
        analysis_str = analysis_from_last_round.get('decision_summary', 'Analysis available but summary is missing.')

    peer_history_str = "No peers in this cluster have run yet in this epoch."
    if peer_history:
        peer_lines = []
        for peer_run in peer_history:
            hps_str = ", ".join(f"{k}={v}" for k, v in peer_run.get('hps_used', {}).items())
            peer_lines.append(f"- Client {peer_run['client_id']} used HPs ({hps_str}) --> {peer_run['result_and_decision']}")
        peer_history_str = "\n".join(peer_lines)

    # ============= NEW: EXTRACT EXACT CONSTRAINTS =============
    client_constraints = []
    server_constraints = []
    
    for param, config in search_space.get('client_hps', {}).items():
        if config.get('type') == 'choice':
            client_constraints.append(f"  - {param}: MUST be one of {config['values']}")
        elif config.get('type') in ['float', 'int']:
            client_constraints.append(f"  - {param}: MUST be between {config['min']} and {config['max']}")
    
    for param, config in search_space.get('server_hps', {}).items():
        if config.get('type') == 'choice':
            server_constraints.append(f"  - {param}: MUST be one of {config['values']}")
        elif config.get('type') in ['float', 'int']:
            server_constraints.append(f"  - {param}: MUST be between {config['min']} and {config['max']}")

    client_constraints_str = "\n".join(client_constraints)
    server_constraints_str = "\n".join(server_constraints)
    
    mu_config = search_space.get('mu', {})
    mu_constraint = f"MUST be between {mu_config.get('min', 0.0)} and {mu_config.get('max', 1.0)}" if mu_config else "any float"

    # Create example with ACTUAL values from search space
    example_hps_dict = {
        "reasoning": "A detailed, multi-part reason for all HP choices.",
        "hps": {
            "client": {
                "learning_rate": search_space.get('client_hps', {}).get('learning_rate', {}).get('initial', 0.001),
                "weight_decay": search_space.get('client_hps', {}).get('weight_decay', {}).get('initial', 5e-05),
                "momentum": search_space.get('client_hps', {}).get('momentum', {}).get('initial', 0.9),
                "optimizer": search_space.get('client_hps', {}).get('optimizer', {}).get('initial', 'AdamW'),
                "scheduler": search_space.get('client_hps', {}).get('scheduler', {}).get('initial', 'CosineAnnealingLR'),
                "local_epochs": search_space.get('client_hps', {}).get('local_epochs', {}).get('initial', 1),
                "batch_size": search_space.get('client_hps', {}).get('batch_size', {}).get('initial', 32),
                "dropout_rate": search_space.get('client_hps', {}).get('dropout_rate', {}).get('initial', 0.1)
            },
            "server": {
                "learning_rate": search_space.get('server_hps', {}).get('learning_rate', {}).get('initial', 0.001),
                "momentum": search_space.get('server_hps', {}).get('momentum', {}).get('initial', 0.9),
                "optimizer": search_space.get('server_hps', {}).get('optimizer', {}).get('initial', 'AdamW'),
                "scheduler": search_space.get('server_hps', {}).get('scheduler', {}).get('initial', 'None')
            },
            "mu": search_space.get('mu', {}).get('initial', 0.01)
        }
    }
    example_json_str = json.dumps(example_hps_dict, indent=4)

    # 1. Create a dynamic, descriptive string for the task.
    task_description = f"Model: {model_name} on {dataset_name}"
    guidance_block = ""
    if model_name.lower() in ['charlstm', 'bert']:
        task_description += " (a **character-level text prediction** task)."
        guidance_block = """
            **EXPERT GUIDANCE FOR TEXT MODELS:**
            - **Primary Goal:** Control overfitting. The gap between train and test accuracy is the most important signal.
            - **Your Best Tool:** `dropout_rate` is your most powerful tool against overfitting in LSTMs. If test accuracy is much lower than train accuracy, your first instinct should be to suggest a HIGHER `dropout_rate`.
            - **Learning Rate:** Text models are sensitive. Lower learning rates (e.g., 1e-4 to 5e-4) are often safer and more stable.
            - **Local Epochs:** More epochs can lead to severe overfitting on the client's small dataset. Be cautious about suggesting more than 1 or 2 epochs unless the model is clearly underfitting.

            """
    else:
        task_description += " (an **image classification** task)."
        guidance_block = """
            **GUIDANCE FOR IMAGE MODELS (CNN/ResNet):**
            - **Key Hyperparameters:** `learning_rate` and `optimizer` choice are often most impactful. `weight_decay` can help with regularization.
            - **Overfitting:** If training accuracy is high but test accuracy is low, suggest a HIGHER `weight_decay` or a LOWER `learning_rate`.
            - **Underfitting:** If both accuracies are low, suggest a HIGHER `learning_rate`.
            """

    return f"""
You are an expert ML engineer and strategist. Your task is to act as a methodical scientist, analyzing all available data to suggest the optimal set of hyperparameters for both the client and the server.

**YOUR THOUGHT PROCESS:**
1.  **Analyze the Task:** First, identify the task (`{task_description}`). Your strategy for text prediction should be different from image classification.
2.  **Review History:** Examine the client's own history and peer history. Is there a pattern? Are high learning rates consistently failing? Is dropout helping?
3.  **Consult Guidance:** Read the expert guidance below and prioritize your choices based on it.
4.  **Formulate Reasoning:** In the "reasoning" key, explain *why* you are choosing each value, referencing the history and guidance. For example: "The test accuracy (45%) is much lower than the train accuracy (70%), indicating overfitting. As per the guidance, I will increase the dropout_rate to 0.4 and suggest a more conservative learning_rate."
5.  **Construct Output:** Provide a single, valid JSON object according to the format, ensuring all values adhere to the STRICT CONSTRAINTS.


{'-'*40}
**OVERALL CONTEXT**
- **Client ID:** {client_id}
- **Model:** {model_name} on {dataset_name}
- **Task:** {task_description}
- **Federated Scheme:** SplitFed with FedProx regularization (controlled by `mu`).

{'-'*40}
{guidance_block} 

{'-'*40}
**CLIENT-SIDE CONTEXT & INSTRUCTIONS**
- **Client Capacity:** {_get_cluster_capacity_string(cluster_id)}. Low-resource clients may need smaller `batch_size`, fewer `local_epochs`, or lower `learning_rate`.
- **Client's Own History:**
{history_str}
- **Peer History (This Epoch):**
{peer_history_str}
- **Last Round's Analysis:** {analysis_str}

{'-'*40}
**STRICT CONSTRAINTS - YOU MUST FOLLOW THESE EXACTLY:**

**Client Parameters:**
{client_constraints_str}

**Server Parameters:**
{server_constraints_str}

**Global Parameter:**
- mu: {mu_constraint}

{'-'*40}
**CRITICAL RULES:**
1. **DO NOT suggest values outside the allowed ranges/choices**
2. **DO NOT be creative with batch sizes** - only use the exact values listed
3. **DO NOT suggest optimizers not in the list**
4. **STICK TO THE CONSTRAINTS** - they exist for a reason

{'-'*40}
**OUTPUT FORMAT & INSTRUCTIONS**
- Return a single, valid JSON object with "reasoning" and "hps" keys.
- The "hps" object must contain "client", "server", and "mu" keys as shown in the example.

⚠️ FINAL INSTRUCTION:
-Output MUST be a single valid JSON object exactly matching the requested keys.
-Do not include any explanations outside the JSON.
-Do not use Markdown or code fences.
-Do not include any introductory or concluding text.

**EXAMPLE OUTPUT (using actual constraint values):**
{example_json_str}

⚠️ YOUR RESPONSE MUST BE PURE JSON.
"""



def get_analysis_prompt(
    client_id: int, cluster_id: int, model_name: str, dataset_name: str,
    results: dict, current_hps: dict, search_space: dict,
    global_epoch: int, local_epochs: int
) -> str:
    
    results_str = f"Final Test Accuracy = {results.get('test_acc', [0.0])[-1]:.2f}%"

    # =================== START OF THE FULLY UPDATED LOGIC ===================

    # 1. Dynamically identify which parameters are choice vs numerical from the search space.
    client_params_config = search_space.get('client_hps', {})
    server_params_config = search_space.get('server_hps', {})
    
    client_choice_params = [k for k, v in client_params_config.items() if v.get('type') == 'choice']
    client_numerical_params = [k for k, v in client_params_config.items() if v.get('type') in ['int', 'float']]
    
    server_choice_params = [k for k, v in server_params_config.items() if v.get('type') == 'choice']
    server_numerical_params = [k for k, v in server_params_config.items() if v.get('type') in ['int', 'float']]

    # 2. Prepare the task description and guidance block (your existing logic).
    task_description = f"Model: {model_name} on {dataset_name}"
    guidance_block = ""
    if model_name.lower() in ['charlstm', 'bert']:
        task_description += " (a **character-level text prediction** task)."
        guidance_block = """

**STRATEGIC GUIDANCE FOR ANALYSIS:**
- **Your Goal:** Your job is to act like a researcher guiding an experiment. You must prune the search space to eliminate bad regions and focus on promising ones.
- **If Overfitting Persists:** If multiple rounds show a large gap between train and test accuracy, the current search space is too aggressive. Your actions should make the search space more conservative. For example: `{"param": "learning_rate", "key": "max", "value": 0.001}` or `{"param": "dropout_rate", "key": "min", "value": 0.3}`.
- **If Performance Stagnates:** If accuracy is stuck, the search space might be too narrow. Consider actions that slightly expand a key parameter range to encourage exploration.

**ANALYSIS GUIDANCE FOR TEXT MODELS:**
- If the model is overfitting (high train acc, low test acc), consider actions that restrict the search space for `learning_rate` (e.g., lower the 'max') or `dropout_rate` (e.g., increase the 'max').
- For numerical parameters like `local_epochs`, you must modify its 'min' or 'max', not 'values'.
"""
    else:
        task_description += " (an **image classification** task)."
        guidance_block = """
**ANALYSIS GUIDANCE FOR IMAGE MODELS:**
- If the model is overfitting (high train acc, low test acc), consider actions that lower the `learning_rate` search space or increase the `weight_decay` search space.
"""
    
    # 3. Construct the final, more explicit prompt.
    return f"""
You are an expert ML HPO analysis agent.

**TASK:**
Analyze the client's performance and provide a list of actions to *refine the hyperparameter search space for future rounds*.
Your actions should be strategic, aiming to guide the hyperparameter search towards better solutions over time.
Return a JSON object with "reasoning" and "actions" keys.


**CONTEXT:**
- Client: {client_id} (Epoch: {global_epoch + 1}, Capacity: {_get_cluster_capacity_string(cluster_id)})
- Task: {task_description}
- HPs Used: {json.dumps(current_hps)}
- Result: {results_str}

{guidance_block}

**CRITICAL INSTRUCTIONS FOR ACTIONS:**

1.  **You MUST use the correct modification key for each parameter based on its type.**
    -   **For CHOICE parameters (use "key": "values"):**
        -   Client: `{client_choice_params}`
        -   Server: `{server_choice_params}`
    -   **For NUMERICAL parameters (use "key": "min" or "max"):**
        -   Client: `{client_numerical_params}`
        -   Server: `{server_numerical_params}`

2.  **Parameter Names**: Use ONLY the exact names from the lists above.
    -   For client parameters, set "target": "client_hps".
    -   For server parameters, set "target": "server_hps".

3.  **Value Formatting**: The "value" field format MUST match the key.
    -   If "key" is "values", then "value" MUST be a LIST (e.g., `[16, 32]`).
    -   If "key" is "min" or "max", then "value" MUST be a single NUMBER (e.g., `0.005`).

**VALID EXAMPLE:**
{{
    "reasoning": "Low accuracy suggests overfitting. Reducing batch size options and lowering max learning rate.",
    "actions": [
        {{
            "param": "batch_size",
            "key": "values",
            "value": [8, 16],
            "target": "client_hps"
        }},
        {{
            "param": "learning_rate",
            "key": "max",
            "value": 0.001,
            "target": "client_hps"
        }}
    ]
}}

⚠️ FINAL INSTRUCTION:
-Output MUST be a single valid JSON object exactly matching the requested keys.
-Do not include any explanations outside the JSON.
-Do not use Markdown or code fences.
-Do not include any introductory or concluding text.

⚠️ YOUR RESPONSE MUST BE PURE JSON.


**YOUR JSON OUTPUT:**
"""

def get_correction_prompt(original_prompt: str, bad_response: str, error_message: str) -> str:
    """
    Generates a prompt to ask the LLM to correct its previous invalid output.
    This function is already general and needs no changes.
    """
    # This prompt is already robust and does not need to be changed.
    return f"""
You are a helpful AI assistant. Your previous response was not correctly formatted, which caused an error. You must try again and strictly follow the output format instructions.

---
**Original Request:**
{original_prompt}
---
**Your Previous Invalid Response:**
{bad_response}
---
**The Error Caused by Your Response:**
{error_message}
---

**Instruction:**
Please try again. Generate a new response that strictly adheres to the requested JSON format and schema from the original request. Do not include any extra text, explanations, or markdown code blocks.
"""