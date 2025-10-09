# code/agent/hp_agent.py

import json
import random
import re
from typing import Dict, Tuple, Any, Optional

from .prompts import get_hp_suggestion_prompt
from .llm_api import call_llm, policy_update
from . import shared_state

import inspect
import os

class HPAgent:
    def __init__(self):
        # Metrics you can inspect or print at the end of training
        self.metrics: Dict[str, int] = {
            "hp_calls": 0,                # total times suggest() was called
            "hp_success": 0,              # times JSON parsed & validated successfully
            "hp_json_errors": 0,          # times JSON parsing failed
            "hp_fallback_used": 0,        # times we fell back to initial defaults
            "clamps": 0,                  # numeric values clamped to [min,max]
            "invalid_choice_fixed": 0,    # invalid 'choice' auto-corrected
            "unknown_param_ignored": 0,   # params not in search space (passed through)
        }
        #New fields to store last prompt/response
        self._last_prompt_by_client: Dict[int, str] = {}
        self._last_response_by_client: Dict[int, str] = {}
        # self.last_prompt: str = ""
        # self.last_response: str = ""
        # Staging area for per-client feedback injected by workflow before suggest(...)
        # Expect dict: {"reward": float, "lyapunov_pass": bool}
        self._pending_feedback: Optional[Dict[str, Any]] = None



    # -------------------- metrics helpers --------------------

    def get_stats(self) -> Dict[str, int]:
        """Return a shallow copy of current counters."""
        return dict(self.metrics)

    def reset_stats(self) -> None:
        """Reset all counters to zero."""
        for k in self.metrics:
            self.metrics[k] = 0

    #New : Tiny hook to accept feedback from workflow
    def attach_feedback(self, *, reward: float, lyapunov_pass: bool) -> None:
        """
        Called by workflow BEFORE suggest(...).
        Stores per-client feedback so suggest(...) can do a tiny on-the-fly adapter update
        using last_prompt/last_response from the previous round.
        """
        self._pending_feedback = {
            "reward": float(reward),
            "lyapunov_pass": bool(lyapunov_pass),
        }

    # -------------------- validation helpers --------------------

    def _validate_hps(self, hps: dict, search_space: dict, hp_key: str) -> dict:
        """
        Validate and sanitize nested HP dicts for a section ('client' or 'server').

        - Clamps numeric params into [min, max]; increments 'clamps' if adjusted.
        - Replaces invalid 'choice' with a random valid value; increments 'invalid_choice_fixed'.
        - If a suggested param isn't in the declared search space section, it is passed through
          (preserving your existing behavior) but counted as 'unknown_param_ignored'.
        """
        validated_hps: Dict[str, Any] = {}
        space_key = f"{hp_key}_hps"

        if hp_key not in hps or space_key not in search_space:
            return {}

        for hp, value in hps[hp_key].items():
            if hp in search_space[space_key]:
                config = search_space[space_key][hp]
                ptype = config.get("type")

                if ptype in ["float", "int"]:
                    # robust float cast; if it fails, clamp to min
                    try:
                        num_value = float(value)
                    except Exception:
                        num_value = float(config.get("min", 0.0))

                    lo = config.get("min", num_value)
                    hi = config.get("max", num_value)
                    clamped = max(lo, min(hi, num_value))
                    if clamped != num_value:
                        print(f"  - WARNING: Clamped '{hp_key}.{hp}' from {value} to {clamped}")
                        self.metrics["clamps"] += 1

                    if ptype == "int":
                        clamped = int(round(clamped))
                    validated_hps[hp] = clamped

                elif ptype == "choice":
                    valid_choices = config.get("values", [])
                    # accept either exact match or string-equivalent match
                    if value not in valid_choices and str(value) not in [str(c) for c in valid_choices]:
                        if valid_choices:
                            replacement = random.choice(valid_choices)
                            print(
                                f"  - WARNING: Invalid choice for '{hp_key}.{hp}'. "
                                f"Got '{value}', using '{replacement}'"
                            )
                            self.metrics["invalid_choice_fixed"] += 1
                            validated_hps[hp] = replacement
                        else:
                            # no declared choices—keep as-is
                            validated_hps[hp] = value
                    else:
                        # pick the value from valid_choices to preserve original type (int vs str)
                        try:
                            correct_value = next(c for c in valid_choices if str(c) == str(value))
                            validated_hps[hp] = correct_value
                        except StopIteration:
                            validated_hps[hp] = valid_choices[0] if valid_choices else value
                else:
                    # unknown/opaque type: keep as-is
                    validated_hps[hp] = value
            else:
                # Param not declared in search space; keep behavior but count it
                self.metrics["unknown_param_ignored"] += 1
                validated_hps[hp] = value

        return validated_hps

    # -------------------- main API --------------------

    def suggest(
        self,
        client_id: int,
        cluster_id: int,
        model_name: str,
        dataset_name: str,
        hpo_report: dict,
        search_space: dict,
        analysis_from_last_round: Optional[dict] = None,
        peer_history: Optional[list] = None,
        arc_cfg: int = 0,
        total_layers: int = 0,
    ) -> Tuple[dict, dict]:
        """
        Ask the LLM for HPs, parse and validate them, and return (final_hps, token_usage).

        Metrics updated here:
          - hp_calls: incremented every time
          - hp_success / hp_json_errors
          - hp_fallback_used (when JSON invalid or 'hps' missing)
          - clamps / invalid_choice_fixed / unknown_param_ignored are updated in _validate_hps
        """

        # --- WHO called me? (file:line) ---
        caller = inspect.stack()[1]
        caller_file = os.path.basename(caller.filename)
        caller_line = caller.lineno
        print(f"[TRACE] HPAgent.suggest called from {caller_file}:{caller_line} (client_id={client_id})")
        # Optional: also stuff this into the event log so it’s visible in aggregates
        try:
            from . import shared_state
            shared_state.log_event("hp", {
                "hp_calls": 1, "hp_success": 0, "hp_json_errors": 0, "hp_fallback_used": 0,
                "clamps": 0, "invalid_choice_fixed": 0, "unknown_param_ignored": 0,
                "caller_file": caller_file, "caller_line": caller_line, "client_id": client_id
            })
        except Exception:
            pass

        # --- PER-CLIENT: load any pending feedback and last I/O ---
        fb = getattr(self, "_pending_feedback", None)

        # Try to fetch last I/O for THIS client
        last_prompt = self._last_prompt_by_client.get(client_id, "")
        last_response = self._last_response_by_client.get(client_id, "")

        # Optional: also fall back to shared_state if agent instances differ
        try:
            io_map = getattr(shared_state, "POLICY_IO", {})
            prev = io_map.get(client_id)
            if prev:
                if not last_prompt:
                    last_prompt = prev.get("prompt", "") or last_prompt
                if not last_response:
                    last_response = prev.get("response", "") or last_response
        except Exception:
            pass

        print(f"[LoRA Preconditions] client={client_id} "
            f"fb={'Y' if bool(fb) else 'N'} "
            f"last_prompt={'Y' if bool(last_prompt) else 'N'} "
            f"last_response={'Y' if bool(last_response) else 'N'}")

        # Tiny on-the-fly update IF we have prior I/O for THIS client
        if fb and last_prompt and last_response:
            try:
                _info = policy_update(
                    prompt=last_prompt,
                    response=last_response,
                    reward=fb["reward"],
                    lyapunov_pass=fb["lyapunov_pass"],
                )
                print(f"[LoRA Update] client={client_id} reward={fb['reward']:.4f} "
                    f"lyapunov={fb['lyapunov_pass']} info={_info}")
            except Exception as e:
                print(f"[LoRA Update] client={client_id} ERROR: {e}")
            finally:
                self._pending_feedback = None



        self.metrics["hp_calls"] += 1

        prompt = get_hp_suggestion_prompt(
            client_id=client_id,
            cluster_id=cluster_id,
            model_name=model_name,
            dataset_name=dataset_name,
            hpo_report=hpo_report,
            search_space=search_space,
            analysis_from_last_round=analysis_from_last_round,
            peer_history=peer_history,
            arc_cfg=arc_cfg,
            total_layers=total_layers,
        )
        self._last_prompt_by_client[client_id] = prompt
        response_json_str, token_usage = call_llm(prompt)

        print("\n" + "---" * 20)
        print(f"<<< RESPONSE FROM HP AGENT (Client {client_id}):")
        print(response_json_str)
        print("---" * 20 + "\n")

        # =============== ROBUST JSON PARSING ===============
        try:
            cleaned = (response_json_str or "").strip()
            if not cleaned:
                raise ValueError("Empty response from LLM")

            # Handle markdown code fences if any slipped through
            if "```json" in cleaned:
                start = cleaned.find("```json") + 7
                end = cleaned.find("```", start)
                if end != -1:
                    cleaned = cleaned[start:end].strip()
            elif "```" in cleaned:
                start = cleaned.find("```") + 3
                end = cleaned.find("```", start)
                if end != -1:
                    cleaned = cleaned[start:end].strip()

            # Strip control chars that might break JSON
            cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", cleaned)

            # Remember response for THIS client for next round
            self._last_response_by_client[client_id] = cleaned

            # Optional: share via shared_state so other instances (if any) can read it
            try:
                if not hasattr(shared_state, "POLICY_IO"):
                    shared_state.POLICY_IO = {}
                shared_state.POLICY_IO[client_id] = {
                    "prompt": self._last_prompt_by_client.get(client_id, ""),
                    "response": cleaned,
                }
            except Exception:
                pass


            data = json.loads(cleaned)
            # optional: could log reasoning if you like
            # reasoning = data.get("reasoning", "No reasoning provided.")
            suggested_hps = data.get("hps", {})

            if not isinstance(suggested_hps, dict) or not suggested_hps:
                raise ValueError("Response 'hps' key is not a valid, non-empty dictionary.")

            final_hps = {
                "client": self._validate_hps(suggested_hps, search_space, "client"),
                "server": self._validate_hps(suggested_hps, search_space, "server"),
                "mu": suggested_hps.get("mu", 0.0),
            }

            # Clamp mu as well (if present in the search space)
            mu_cfg = search_space.get("mu", {})
            if mu_cfg:
                lo = mu_cfg.get("min", 0.0)
                hi = mu_cfg.get("max", 1.0)
                try:
                    mu_val = float(final_hps["mu"])
                except Exception:
                    mu_val = float(lo)
                mu_clamped = max(lo, min(hi, mu_val))
                if mu_clamped != mu_val:
                    print(f"  - WARNING: Clamped 'mu' from {mu_val} to {mu_clamped}")
                    self.metrics["clamps"] += 1
                final_hps["mu"] = mu_clamped

            self.metrics["hp_success"] += 1
            # snapshot to shared state so you can print at end of training
            shared_state.HP_AGENT_STATS = self.get_stats()
            shared_state.save_stats("hp", self.get_stats()) 
            return final_hps, token_usage

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Count the error and fall back to initial values
            self.metrics["hp_json_errors"] += 1
            self.metrics["hp_fallback_used"] += 1

            print(f"Error: Could not parse HPs from LLM response: {e}")
            snippet = (response_json_str or "")[:200]
            print(f"   Raw response (first 200 chars): {repr(snippet)}")
            print(f"   Using fallback hyperparameters for Client {client_id}")

            fallback_hps = {
                "client": {k: v["initial"] for k, v in search_space.get("client_hps", {}).items()},
                "server": {k: v["initial"] for k, v in search_space.get("server_hps", {}).items()},
                "mu": search_space.get("mu", {}).get("initial", 0.0),
            }

            # Best-effort remember even if JSON failed (helps the next round)
            clean_try = (response_json_str or "").strip()
            self._last_response_by_client[client_id] = clean_try
            try:
                if not hasattr(shared_state, "POLICY_IO"):
                    shared_state.POLICY_IO = {}
                shared_state.POLICY_IO[client_id] = {
                    "prompt": self._last_prompt_by_client.get(client_id, ""),
                    "response": clean_try,
                }
            except Exception:
                pass


            # snapshot to shared state so you can print at end of training
            shared_state.HP_AGENT_STATS = self.get_stats()
            shared_state.save_stats("hp", self.get_stats()) 
            return fallback_hps, token_usage
