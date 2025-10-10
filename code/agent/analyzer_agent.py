# code/agent/analyzer_agent.py

import json
import copy
from typing import Dict, Any, List, Tuple

from .prompts import get_analysis_prompt
from .llm_api import call_llm
from . import shared_state


class AnalyzerAgent:
    """
    Gets high-level analysis from an LLM, then programmatically updates the search space.
    Tracks detailed metrics and persists them to shared_state so you can aggregate later.
    """

    def __init__(self):
        # Cumulative metrics (persisted via shared_state.save_stats)
        self.metrics: Dict[str, int] = {
            "analyze_calls": 0,
            "analyze_success": 0,
            "analyze_json_errors": 0,

            "actions_total": 0,                 # actions considered after normalization
            "actions_applied": 0,               # actually applied to search space

            # skip reasons
            "actions_skipped_malformed": 0,     # malformed dict or missing keys
            "actions_skipped_invalid_target": 0,# bad target or param not found
            "actions_skipped_invalid_key": 0,   # key not valid for that param
            "actions_values_list_type_error": 0,# choice param 'values' not list

            # auto-fixes
            "actions_numeric_values_coerced": 0 # numeric 'values' list -> min/max
        }

    # -------- metrics helpers --------
    def get_stats(self) -> Dict[str, int]:
        return dict(self.metrics)

    def reset_stats(self) -> None:
        for k in self.metrics:
            self.metrics[k] = 0

    # -------- internal helpers --------

    @staticmethod
    def _numeric_param_sets(search_space: dict) -> Tuple[set, set]:
        """
        Return sets of numeric param names for client/server based on search_space types.
        """
        client_cfg = search_space.get("client_hps", {})
        server_cfg = search_space.get("server_hps", {})
        numeric_client = {k for k, v in client_cfg.items() if v.get("type") in ("int", "float")}
        numeric_server = {k for k, v in server_cfg.items() if v.get("type") in ("int", "float")}
        return numeric_client, numeric_server

    @staticmethod
    def _param_conf(search_space: dict, target: str, param: str) -> dict | None:
        return search_space.get(target, {}).get(param)

    @staticmethod
    def _is_choice_param(pconf: dict) -> bool:
        return pconf and pconf.get("type") == "choice"

    @staticmethod
    def _is_numeric_param(pconf: dict) -> bool:
        return pconf and pconf.get("type") in ("int", "float")

    @staticmethod
    def _coerce_numeric(x, as_int: bool):
        try:
            v = float(x)
            return int(round(v)) if as_int else v
        except Exception:
            return None

    def _normalize_actions(self, search_space: dict, actions: List[dict]) -> List[dict]:
        """
        Normalize/validate actions before application.

        Supports:
        - For numeric params:
            * key: "range", value: [min, max]  -> expands to two actions (min, max)
            * key: "values", value: [a, b, ...] -> coerces to range(min(values), max(values))
        - For choice params:
            * key: "values", value: list -> subset of original allowed list (preserve original types)

        Increments metrics for skips and coercions.
        """
        if not isinstance(actions, list):
            return []

        norm: List[dict] = []
        for a in actions:
            if not (isinstance(a, dict) and all(k in a for k in ("param", "key", "value", "target"))):
                print(f"  - WARNING: Malformed action object skipped: {a}")
                self.metrics["actions_skipped_malformed"] += 1
                continue

            target = a["target"]
            param = a["param"]
            key = a["key"]
            value = a["value"]

            if target not in ("client_hps", "server_hps"):
                print(f"  - WARNING: Invalid target '{target}'. Action skipped.")
                self.metrics["actions_skipped_invalid_target"] += 1
                continue

            pconf = self._param_conf(search_space, target, param)
            if not pconf:
                print(f"  - WARNING: Invalid target/param '{target}/{param}'. Action skipped.")
                self.metrics["actions_skipped_invalid_target"] += 1
                continue

            # Choice param: only 'values' is valid, and must be list subset of original
            if self._is_choice_param(pconf):
                if key != "values":
                    print(f"  - WARNING: Invalid key '{key}' for choice param '{param}'. Skipped.")
                    self.metrics["actions_skipped_invalid_key"] += 1
                    continue
                if not isinstance(value, list):
                    print(f"  - WARNING: 'values' for choice param '{param}' must be a list. Skipped.")
                    self.metrics["actions_values_list_type_error"] += 1
                    continue

                original = pconf.get("values", [])
                # Preserve original types while matching by string
                allowed_str = {str(x): x for x in original}
                subset = []
                for item in value:
                    s = str(item)
                    if s in allowed_str:
                        subset.append(allowed_str[s])

                if not subset:
                    print(f"  - WARNING: No valid choices left for '{param}'. Skipped.")
                    self.metrics["actions_skipped_invalid_key"] += 1
                    continue

                norm.append({"param": param, "key": "values", "value": subset, "target": target})
                continue

            # Numeric param
            if self._is_numeric_param(pconf):
                as_int = (pconf.get("type") == "int")

                # convenience: "range": [min, max]
                if key == "range":
                    if not (isinstance(value, (list, tuple)) and len(value) == 2):
                        print(f"  - WARNING: 'range' requires [min, max] for '{param}'. Skipped.")
                        self.metrics["actions_skipped_malformed"] += 1
                        continue

                    lo = self._coerce_numeric(value[0], as_int)
                    hi = self._coerce_numeric(value[1], as_int)
                    if lo is None or hi is None:
                        print(f"  - WARNING: Could not coerce range bounds for '{param}'. Skipped.")
                        self.metrics["actions_skipped_malformed"] += 1
                        continue
                    if lo > hi:
                        lo, hi = hi, lo

                    # clamp within original bounds if available
                    if isinstance(pconf.get("min"), (int, float)):
                        lo = max(self._coerce_numeric(pconf["min"], as_int), lo)
                    if isinstance(pconf.get("max"), (int, float)):
                        hi = min(self._coerce_numeric(pconf["max"], as_int), hi)
                    if lo > hi:
                        print(f"  - WARNING: Normalized range inverted for '{param}' after clamping. Skipped.")
                        self.metrics["actions_skipped_malformed"] += 1
                        continue

                    norm.append({"param": param, "key": "min", "value": lo, "target": target})
                    norm.append({"param": param, "key": "max", "value": hi, "target": target})
                    continue

                # convenience: numeric with 'values' list -> coerce to min/max
                if key == "values" and isinstance(value, list) and len(value) >= 1:
                    lo = min(value)
                    hi = max(value)
                    lo = self._coerce_numeric(lo, as_int)
                    hi = self._coerce_numeric(hi, as_int)
                    if lo is None or hi is None:
                        print(f"  - WARNING: Non-numeric entries in 'values' for '{param}'. Skipped.")
                        self.metrics["actions_skipped_malformed"] += 1
                        continue
                    if lo > hi:
                        lo, hi = hi, lo
                    # clamp within original bounds
                    if isinstance(pconf.get("min"), (int, float)):
                        lo = max(self._coerce_numeric(pconf["min"], as_int), lo)
                    if isinstance(pconf.get("max"), (int, float)):
                        hi = min(self._coerce_numeric(pconf["max"], as_int), hi)
                    if lo > hi:
                        print(f"  - WARNING: Coerced 'values' produced inverted range for '{param}'. Skipped.")
                        self.metrics["actions_skipped_malformed"] += 1
                        continue

                    norm.append({"param": param, "key": "min", "value": lo, "target": target})
                    norm.append({"param": param, "key": "max", "value": hi, "target": target})
                    self.metrics["actions_numeric_values_coerced"] += 1
                    continue

                # canonical numeric keys
                if key not in ("min", "max"):
                    print(f"  - WARNING: Invalid key '{key}' for numeric param '{param}'. Skipped.")
                    self.metrics["actions_skipped_invalid_key"] += 1
                    continue

                v = self._coerce_numeric(value, as_int)
                if v is None:
                    print(f"  - WARNING: Non-numeric value for '{param}.{key}'. Skipped.")
                    self.metrics["actions_skipped_malformed"] += 1
                    continue

                # clamp to original bounds if present
                if key == "min" and isinstance(pconf.get("max"), (int, float)):
                    vmax = pconf["max"]
                    if v > vmax:
                        v = self._coerce_numeric(vmax, as_int)
                if key == "max" and isinstance(pconf.get("min"), (int, float)):
                    vmin = pconf["min"]
                    if v < vmin:
                        v = self._coerce_numeric(vmin, as_int)

                norm.append({"param": param, "key": key, "value": v, "target": target})
                continue

            # Unknown param type
            print(f"  - WARNING: Unknown type for param '{param}'. Skipped.")
            self.metrics["actions_skipped_invalid_target"] += 1

        return norm

    def _apply_actions_to_search_space(self, search_space: dict, actions: list) -> dict:
        """
        Apply normalized actions to a deep-copied search space with final safety checks.
        """
        new_space = copy.deepcopy(search_space)

        # 1) Normalize actions first (adds to skip/coercion counters)
        actions = self._normalize_actions(search_space, actions)
        # 2) Count what we're actually considering now
        self.metrics["actions_total"] += len(actions)

        # 3) Apply
        for action in actions:
            target = action["target"]
            param = action["param"]
            key = action["key"]
            value = action["value"]

            if target not in new_space or param not in new_space[target]:
                print(f"  - WARNING: Invalid target/param '{target}/{param}'. Action skipped.")
                self.metrics["actions_skipped_invalid_target"] += 1
                continue

            # final key check against existing config keys
            if key not in new_space[target][param]:
                print(f"  - WARNING: Invalid key '{key}' for param '{param}'. Action skipped.")
                self.metrics["actions_skipped_invalid_key"] += 1
                continue

            new_space[target][param][key] = value
            self.metrics["actions_applied"] += 1

        return new_space

    # -------- public API --------

    def analyze(
        self,
        client_id: int,
        cluster_id: int,
        model_name: str,
        dataset_name: str,
        results: dict,
        current_hps: dict,
        search_space: dict,
        global_epoch: int,
        local_epochs: int,
    ):
        """
        Calls the LLM, parses actions, normalizes them (supporting 'range'), applies to search space,
        and returns (new_search_space, reasoning_obj, token_usage).
        Also updates metrics and persists snapshots to shared_state for end-of-run aggregation.
        """
        self.metrics["analyze_calls"] += 1

        prompt = get_analysis_prompt(
            client_id=client_id,
            cluster_id=cluster_id,
            model_name=model_name,
            dataset_name=dataset_name,
            results=results,
            current_hps=current_hps,
            search_space=search_space,
            global_epoch=global_epoch,
            local_epochs=local_epochs,
        )

        response_str, token_usage = call_llm(prompt)

        print("\n" + "---" * 20)
        print(f"<<< RESPONSE FROM ANALYZER AGENT (Client {client_id}):")
        print("we skipped printing the full ANALYZER response to avoid clutter")
        # print(response_str)
        print("---" * 20 + "\n")

        try:
            data = json.loads(response_str)
            reasoning = data.get("reasoning", "No reasoning provided by LLM.")
            actions = data.get("actions", [])

            new_search_space = self._apply_actions_to_search_space(search_space, actions)

            # Build a per-call reasoning summary (do not use cumulative counters here)
            applied_this_round = 0  # derive from delta of counters if you like; simpler: recount
            # (Optional) you can recount here, but cumulative 'actions_applied' is OK for now.

            final_reasoning_obj = {
                "performance_summary": reasoning,
                "decision_summary": (
                    f"Applied actions; totals so far — applied: {self.metrics['actions_applied']}, "
                    f"coerced numeric 'values'→range: {self.metrics['actions_numeric_values_coerced']}."
                )
            }

            self.metrics["analyze_success"] += 1

            # Persist a snapshot that the main process can aggregate
            shared_state.ANALYZER_AGENT_STATS = self.get_stats()
            shared_state.save_stats("analyzer", self.get_stats())

            return new_search_space, final_reasoning_obj, token_usage

        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Warning: Analyzer for Client {client_id} could not generate a valid response. Error: {e}. Re-using old search space.")

            self.metrics["analyze_json_errors"] += 1
            shared_state.ANALYZER_AGENT_STATS = self.get_stats()
            shared_state.save_stats("analyzer", self.get_stats())

            return search_space, None, token_usage






# # code/agent/analyzer_agent.py

# import json
# import copy
# from typing import Dict, Any, List, Tuple, Optional

# from .prompts import get_analysis_prompt
# from .llm_api import call_llm
# from . import shared_state


# class AnalyzerAgent:
#     """
#     An agent that gets high-level analysis from an LLM, then uses
#     that analysis to programmatically build the new search space.
#     """

#     def __init__(self):
#         # Metrics for visibility & end-of-run summary
#         self.metrics: Dict[str, int] = {
#             "analyze_calls": 0,                 # total times analyze() called
#             "analyze_success": 0,               # times JSON parsed successfully
#             "analyze_json_errors": 0,           # times JSON parsing failed

#             "actions_total": 0,                 # total actions received (after coercion)
#             "actions_applied": 0,               # actions successfully applied

#             # skip reasons (pre-/post- coercion)
#             "actions_skipped_malformed": 0,     # action missing required keys/shape
#             "actions_skipped_invalid_target": 0,# invalid target/param path
#             "actions_skipped_invalid_key": 0,   # key not valid for that param
#             "actions_values_list_type_error": 0,# choice param 'values' not list

#             # helpful auto-fixes
#             "actions_numeric_values_coerced": 0 # numeric param had 'values' list -> coercion to min/max
#         }

#     # -------- metrics helpers --------
#     def get_stats(self) -> Dict[str, int]:
#         return dict(self.metrics)

#     def reset_stats(self) -> None:
#         for k in self.metrics:
#             self.metrics[k] = 0

#     # -------- internal helpers --------

#     @staticmethod
#     def _numeric_param_sets(search_space: dict) -> Tuple[set, set]:
#         """
#         Return sets of numeric param names for client/server based on search_space types.
#         """
#         client_cfg = search_space.get("client_hps", {})
#         server_cfg = search_space.get("server_hps", {})
#         numeric_client = {k for k, v in client_cfg.items() if v.get("type") in ("int", "float")}
#         numeric_server = {k for k, v in server_cfg.items() if v.get("type") in ("int", "float")}
#         return numeric_client, numeric_server

#     def _coerce_numeric_values_to_range(
#         self, actions: List[dict], search_space: dict
#     ) -> List[dict]:
#         """
#         For numeric params: if an action uses key='values' with a list,
#         convert it into two actions: (min=lo) and (max=hi).
#         Leaves choice params unchanged.
#         """
#         numeric_client, numeric_server = self._numeric_param_sets(search_space)
#         out: List[dict] = []

#         for a in actions:
#             try:
#                 if not isinstance(a, dict):
#                     self.metrics["actions_skipped_malformed"] += 1
#                     continue

#                 param = a.get("param")
#                 key   = a.get("key")
#                 val   = a.get("value")
#                 tgt   = a.get("target")

#                 if tgt not in ("client_hps", "server_hps") or not param or key is None:
#                     self.metrics["actions_skipped_malformed"] += 1
#                     continue

#                 is_numeric = (
#                     (tgt == "client_hps" and param in numeric_client) or
#                     (tgt == "server_hps" and param in numeric_server)
#                 )

#                 # If numeric and got 'values' list, coerce to min/max
#                 if is_numeric and key == "values" and isinstance(val, list) and len(val) >= 1:
#                     lo = min(val)
#                     hi = max(val)
#                     out.append({"param": param, "key": "min", "value": lo, "target": tgt})
#                     out.append({"param": param, "key": "max", "value": hi, "target": tgt})
#                     self.metrics["actions_numeric_values_coerced"] += 1
#                 else:
#                     out.append(a)
#             except Exception:
#                 self.metrics["actions_skipped_malformed"] += 1
#                 continue

#         return out

#     def _apply_actions_to_search_space(self, search_space: dict, actions: list) -> dict:
#         """
#         Safely applies LLM-suggested actions to the correct part of the search space,
#         with strong validation and counters.
#         """
#         new_space = copy.deepcopy(search_space)
#         if not isinstance(actions, list):
#             print("  - WARNING: 'actions' field from LLM is not a list. No changes applied.")
#             return new_space

#         for action in actions:
#             if not (isinstance(action, dict) and all(k in action for k in ("param", "key", "value", "target"))):
#                 print(f"  - WARNING: Malformed action object skipped: {action}")
#                 self.metrics["actions_skipped_malformed"] += 1
#                 continue

#             target_space_key = action["target"]
#             param = action["param"]
#             key_to_change = action["key"]
#             value = action["value"]

#             # Check target/param validity
#             if target_space_key not in new_space or param not in new_space[target_space_key]:
#                 print(f"  - WARNING: Invalid target/param '{target_space_key}/{param}'. Action skipped.")
#                 self.metrics["actions_skipped_invalid_target"] += 1
#                 continue

#             # If 'values' for choice param, enforce list type
#             if key_to_change == "values":
#                 param_type = new_space[target_space_key][param].get("type")
#                 if param_type == "choice" and not isinstance(value, list):
#                     print(f"  - ERROR: Invalid value for '{param}' values. Expected a list but got {type(value)}. Action skipped.")
#                     self.metrics["actions_values_list_type_error"] += 1
#                     continue

#             # Validate the key is present in the param config
#             if key_to_change not in new_space[target_space_key][param]:
#                 print(f"  - WARNING: Invalid key '{key_to_change}' for param '{param}'. Action skipped.")
#                 self.metrics["actions_skipped_invalid_key"] += 1
#                 continue

#             # All checks pass — apply the change
#             new_space[target_space_key][param][key_to_change] = value
#             self.metrics["actions_applied"] += 1

#         return new_space

#     # -------- public API --------

#     def analyze(
#         self,
#         client_id: int,
#         cluster_id: int,
#         model_name: str,
#         dataset_name: str,
#         results: dict,
#         current_hps: dict,
#         search_space: dict,
#         global_epoch: int,
#         local_epochs: int,
#     ):
#         """
#         Calls the LLM to get reasoning and a list of actions,
#         then builds and returns the new search space along with reasoning and token usage.
#         """
#         self.metrics["analyze_calls"] += 1

#         prompt = get_analysis_prompt(
#             client_id=client_id,
#             cluster_id=cluster_id,
#             model_name=model_name,
#             dataset_name=dataset_name,
#             results=results,
#             current_hps=current_hps,
#             search_space=search_space,
#             global_epoch=global_epoch,
#             local_epochs=local_epochs,
#         )

#         response_str, token_usage = call_llm(prompt)

#         print("\n" + "---" * 20)
#         print(f"<<< RESPONSE FROM ANALYZER AGENT (Client {client_id}):")
#         print(response_str)
#         print("---" * 20 + "\n")

#         try:
#             data = json.loads(response_str)
#             reasoning = data.get("reasoning", "No reasoning provided by LLM.")
#             actions = data.get("actions", [])

#             # Coerce illegal 'values' for numeric params into min/max pairs
#             actions = self._coerce_numeric_values_to_range(actions, search_space)

#             # Count total after coercion (what we actually consider)
#             self.metrics["actions_total"] += len(actions)

#             new_search_space = self._apply_actions_to_search_space(search_space, actions)

#             final_reasoning_obj = {
#                 "performance_summary": reasoning,
#                 "decision_summary": f"Applied {self.metrics['actions_applied']} action(s) this round. "
#                                     f"Coerced {self.metrics['actions_numeric_values_coerced']} numeric 'values' actions to ranges."
#             }

#             self.metrics["analyze_success"] += 1
#             # expose snapshot for end-of-run summary
#             shared_state.ANALYZER_AGENT_STATS = self.get_stats()
#             shared_state.save_stats("analyzer", self.get_stats()) 

#             return new_search_space, final_reasoning_obj, token_usage

#         except (json.JSONDecodeError, AttributeError) as e:
#             print(f"Warning: Analyzer for Client {client_id} could not generate a valid response. Error: {e}. Re-using old search space.")

#             self.metrics["analyze_json_errors"] += 1
#             shared_state.ANALYZER_AGENT_STATS = self.get_stats()
#             shared_state.save_stats("analyzer", self.get_stats()) 

#             return search_space, None, token_usage
