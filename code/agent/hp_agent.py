import json
from .prompts import get_hp_suggestion_prompt
from .llm_api import call_llm
import random

class HPAgent:

    def _validate_hps(self, hps: dict, search_space: dict, hp_key: str):
        """Helper to validate a nested HP dictionary (e.g., 'client' or 'server')."""
        validated_hps = {}
        space_key = f"{hp_key}_hps"
        
        if hp_key not in hps or space_key not in search_space:
            return {}

        for hp, value in hps[hp_key].items():
            if hp in search_space[space_key]:
                config = search_space[space_key][hp]
                param_type = config.get('type')

                if param_type in ['float', 'int']:
                    # Clamp the value within the min/max bounds
                    clamped_value = max(config['min'], min(config['max'], float(value))) # Use float() for safety
                    if clamped_value != value:
                        print(f"  - WARNING: Clamped '{hp_key}.{hp}' from {value} to {clamped_value}")
                    validated_hps[hp] = clamped_value
                
                elif param_type == 'choice':
                    # --- FIX: Handle potential type mismatch (e.g., "32" vs 32) ---
                    valid_choices = config.get('values', [])
                    # Try to match by value or by string representation of value
                    if value not in valid_choices and str(value) not in [str(c) for c in valid_choices]:
                        valid_choice = random.choice(valid_choices)
                        print(f"  - WARNING: Invalid choice for '{hp_key}.{hp}'. Got '{value}', using random choice '{valid_choice}'")
                        validated_hps[hp] = valid_choice
                    else:
                        # Return the original correct type from the list, not the LLM's string version
                        try:
                            # Find the matching choice to preserve original type (int vs str)
                            correct_value = [c for c in valid_choices if str(c) == str(value)][0]
                            validated_hps[hp] = correct_value
                        except IndexError: # Should not happen due to the check above, but as a safeguard
                            validated_hps[hp] = random.choice(valid_choices)
                else:
                    validated_hps[hp] = value
            else:
                validated_hps[hp] = value
        return validated_hps



    def suggest(self, client_id, cluster_id, model_name, dataset_name, hpo_report,
            search_space, analysis_from_last_round: dict | None = None,
            peer_history: list | None = None, arc_cfg: int = 0, total_layers: int = 0):
    
        prompt = get_hp_suggestion_prompt(
            client_id=client_id, cluster_id=cluster_id, model_name=model_name,
            dataset_name=dataset_name, hpo_report=hpo_report,
            search_space=search_space, analysis_from_last_round=analysis_from_last_round,
            peer_history=peer_history, arc_cfg=arc_cfg, total_layers=total_layers
        )

        response_json_str, token_usage = call_llm(prompt)

        print("\n" + "---" * 20)
        print(f"<<< RESPONSE FROM HP AGENT (Client {client_id}):")
        print(response_json_str)
        print("---" * 20 + "\n")
        
        # =============== ROBUST JSON PARSING ===============
        try:
            # Clean the response string
            if not response_json_str or not response_json_str.strip():
                raise ValueError("Empty response from LLM")
            
            # Remove common problematic characters
            cleaned_response = response_json_str.strip()
            
            # Try to extract JSON if it's wrapped in markdown code blocks
            if "```json" in cleaned_response:
                start = cleaned_response.find("```json") + 7
                end = cleaned_response.find("```", start)
                if end != -1:
                    cleaned_response = cleaned_response[start:end].strip()
            elif "```" in cleaned_response:
                start = cleaned_response.find("```") + 3
                end = cleaned_response.find("```", start)
                if end != -1:
                    cleaned_response = cleaned_response[start:end].strip()
            
            # Remove control characters that break JSON
            import re
            cleaned_response = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_response)
            
            response_data = json.loads(cleaned_response)
            llm_reasoning = response_data.get("reasoning", "No reasoning provided.")
            suggested_hps = response_data.get("hps", {})
            
            if not isinstance(suggested_hps, dict) or not suggested_hps:
                raise ValueError("Response 'hps' key is not a valid, non-empty dictionary.")
            
            # print(f"--- [HP Agent Verdict for Client {client_id}] ---")
            # print(f"  - Reasoning: {llm_reasoning}")
            #print(f"LLM Suggested HPs (raw): {json.dumps(suggested_hps, indent=2)}")

            # Existing validation logic (unchanged)
            final_hps = {
                "client": self._validate_hps(suggested_hps, search_space, 'client'),
                "server": self._validate_hps(suggested_hps, search_space, 'server'),
                "mu": suggested_hps.get('mu', 0.0)
            }
            
            # Clamp mu as well
            mu_config = search_space.get('mu', {})
            if mu_config:
                final_hps['mu'] = max(mu_config.get('min', 0.0), min(mu_config.get('max', 1.0), final_hps['mu']))

            #print(f"Final suggested HPs (validated): {json.dumps(final_hps, indent=2)}")
            #print("---")
            return final_hps, token_usage
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"❌ Error: Could not parse HPs from LLM response: {e}")
            print(f"   Raw response (first 200 chars): {repr(response_json_str[:200])}")
            
            # Fallback to initial values for the full nested structure
            print(f"   Using fallback hyperparameters for Client {client_id}")
            fallback_hps = {
                "client": {k: v['initial'] for k, v in search_space.get('client_hps', {}).items()},
                "server": {k: v['initial'] for k, v in search_space.get('server_hps', {}).items()},
                "mu": search_space.get('mu', {}).get('initial', 0.0)
            }
            return fallback_hps, token_usage