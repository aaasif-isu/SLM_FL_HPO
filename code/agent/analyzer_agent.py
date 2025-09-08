# In code/agent/analyzer_agent.py

import json
import copy
from .prompts import get_analysis_prompt
from .llm_api import call_llm

class AnalyzerAgent:
    """
    An agent that gets high-level analysis from an LLM, then uses
    that analysis to programmatically build the new search space.
    """

    def _apply_actions_to_search_space(self, search_space: dict, actions: list) -> dict:
        """
        Safely applies the LLM's suggested actions to the correct part of the search space,
        with strong validation to prevent corruption.
        """
        new_space = copy.deepcopy(search_space)
        if not isinstance(actions, list):
            print("  - WARNING: 'actions' field from LLM is not a list. No changes applied.")
            return new_space

        for action in actions:
            if not (isinstance(action, dict) and all(k in action for k in ['param', 'key', 'value', 'target'])):
                print(f"  - WARNING: Malformed action object skipped: {action}")
                continue

            target_space_key = action['target']
            param = action['param']
            key_to_change = action['key']
            value = action['value']
            
            # Check if the target and param are valid before proceeding
            if target_space_key not in new_space or param not in new_space[target_space_key]:
                print(f"  - WARNING: Invalid target/param '{target_space_key}/{param}'. Action skipped.")
                continue

            # --- THIS IS THE CRITICAL FIX ---
            # If the LLM wants to change the 'values' of a choice parameter,
            # we MUST ensure the new value is a list.
            if key_to_change == 'values':
                param_type = new_space[target_space_key][param].get('type')
                if param_type == 'choice' and not isinstance(value, list):
                    print(f"  - ERROR: Invalid value for '{param}' values. Expected a list but got {type(value)}. Action skipped.")
                    continue # Skip this invalid action to prevent corrupting the search space

            # Check if the key itself is valid for that parameter
            if key_to_change not in new_space[target_space_key][param]:
                print(f"  - WARNING: Invalid key '{key_to_change}' for param '{param}'. Action skipped.")
                continue
            
            # If all checks pass, apply the change
            #print(f"  - Applying action: Setting {target_space_key}.{param}.{key_to_change} = {value}")
            new_space[target_space_key][param][key_to_change] = value
                
        return new_space

    def analyze(self, client_id, cluster_id, model_name, dataset_name, results, current_hps, search_space, global_epoch, local_epochs):
        """
        Calls the LLM to get reasoning and a list of actions, then builds the
        new search space and returns it along with the reasoning.
        """
        prompt = get_analysis_prompt(
            client_id=client_id, cluster_id=cluster_id, model_name=model_name,
            dataset_name=dataset_name, results=results, current_hps=current_hps,
            search_space=search_space, global_epoch=global_epoch, local_epochs=local_epochs
        )
        
        response_str, token_usage = call_llm(prompt)

        print("\n" + "---" * 20)
        print(f"<<< RESPONSE FROM ANALYZER AGENT (Client {client_id}):")
        print(response_str)
        print("---" * 20 + "\n")
        
        try:
            response_data = json.loads(response_str)
            reasoning = response_data.get("reasoning", "No reasoning provided by LLM.")
            actions = response_data.get("actions", [])
            
            # print(f"\n--- [Analyzer Reasoning for Client {client_id}] ---")
            # print(f"  - LLM Reasoning: {reasoning}")
            
            new_search_space = self._apply_actions_to_search_space(search_space, actions)
            
            # print("--- [Proposed New Search Space] ---")
            # print(json.dumps(new_search_space, indent=2))
            # print("-" * 45)

            final_reasoning_obj = {
                "performance_summary": reasoning,
                "decision_summary": f"Applied {len(actions)} action(s) to refine the search space."
            }
            return new_search_space, final_reasoning_obj, token_usage

        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Warning: Analyzer for Client {client_id} could not generate a valid response. Error: {e}. Re-using old search space.")
            default_usage = {"prompt_tokens": 0, "completion_tokens": 0}

            return search_space, None, token_usage

