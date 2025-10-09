# code/ssfl/strategies.py

import random
import numpy as np
from abc import ABC, abstractmethod
from ssfl.trainer_utils import train_single_client
from torch.utils.data import DataLoader
import time
from agent.shared_state import results_queue

from agent.hp_agent import HPAgent
from agent import shared_state

class HPOStrategy(ABC):
    def __init__(self, initial_search_space: dict, client_states: list, **kwargs):
        self.initial_search_space = initial_search_space
        self.client_states = client_states
        # NEW: Get HPO patience from the kwargs, which will be passed from the config
        #self.hpo_patience = kwargs.get('hpo_patience', 3) # Default to 3


    @abstractmethod
    def get_hyperparameters(self, context: dict) -> tuple:
        raise NotImplementedError

    def update_persistent_state(self, client_id: int, context: dict, final_state: dict):
        # This base method is now only used by AgentStrategy.
        # It needs to be updated to handle the new "last_analysis" field.
        if self.client_states and client_id < len(self.client_states):
            global_epoch = context['training_args'].get('global_epoch', -1)

            new_report_entry = {
                "hps_suggested": final_state.get('hps', {}),
                "final_test_accuracy": final_state.get('results', {}).get('test_acc', [None])[-1]
            }
            
            self.client_states[client_id]['hpo_report'][global_epoch] = new_report_entry

            # ========== EARLY TRIMMING LOGIC ==========
            hpo_report = self.client_states[client_id]['hpo_report']
            
            # Keep only the most recent epochs within the window
            if len(hpo_report) > self.history_window:
                recent_epochs = sorted(hpo_report.keys())[-self.history_window:]
                
                # Special case: If this client just early stopped, always keep the stopping epoch
                current_epoch = context['training_args'].get('global_epoch', -1)
                if current_epoch not in recent_epochs:
                    # Replace oldest with current (stopping) epoch
                    recent_epochs = recent_epochs[1:] + [current_epoch]

                self.client_states[client_id]['hpo_report'] = {
                    epoch: hpo_report[epoch] for epoch in recent_epochs
                }
                print(f"  - Trimmed Client {client_id} history to {len(recent_epochs)} recent epochs")
        
            self.client_states[client_id]['search_space'] = final_state.get('search_space', self.initial_search_space)
            
            # --- THE FINAL FIX: Save the reasoning for the next round ---
            self.client_states[client_id]['last_analysis'] = final_state.get('last_analysis')

class AgentStrategy(HPOStrategy):
    def __init__(self, initial_search_space: dict, client_states: list, detailed_log_filename: str, **kwargs):
        # Init can remain the same, without the graph
        super().__init__(initial_search_space, client_states, **kwargs)
        self.history_window = kwargs.get('history_window', 5)
        # 2. CREATE AN AGENT INSTANCE FOR INITIAL SUGGESTIONS
        self.hp_agent = HPAgent()
        self.detailed_log_filename = detailed_log_filename
        




    def _get_reasoned_initial_hps(self, context: dict) -> dict:
        """
        Calls the HPAgent directly to get a reasoned set of HPs for the first run,
        mimicking the intelligence of your original framework.
        """
        print(f"  --> Client {context['client_id']}: Getting reasoned initial HPs from LLM (first run)...")
        # This call has no performance history, but the agent can still reason
        # based on client characteristics and the search space.
        hps, _ = self.hp_agent.suggest(
            client_id=context['client_id'],
            cluster_id=context['cluster_id'],
            model_name=context['model_name'],
            dataset_name=context['dataset_name'],
            hpo_report={}, # No history yet
            search_space=self.client_states[context['client_id']]['search_space'],
            analysis_from_last_round=None, # No analysis yet
            peer_history=context.get('peer_history')
        )
        return hps

    def _get_initial_hps(self, search_space: dict) -> dict:
        """
        Recursively selects simple, concrete default HPs from the initial search space.
        This is a robust method to get a valid "order" from the "menu".
        """
        hps = {}
        for key, value in search_space.items():
            if isinstance(value, dict):
                # If the value is a dictionary defining a parameter (e.g., {'min':...}),
                # extract a single default value.
                if 'default' in value:
                    hps[key] = value['default']
                elif 'min' in value:
                    hps[key] = value['min']  # Use the minimum as a safe default
                else:
                    # Otherwise, it's a nested group of HPs (like 'client' or 'server'). Recurse.
                    hps[key] = self._get_initial_hps(value)
            elif isinstance(value, list):
                # If it's a list of choices, take the first one.
                hps[key] = value[0]
            else:
                # It's already a concrete value.
                hps[key] = value
        return hps


    
    def get_hyperparameters(self, context: dict) -> tuple:
        client_id = context['client_id']
        current_state = self.client_states[client_id]


        # --- THIS IS THE FINAL, CORRECT LOGIC ---
        this_epoch = context['training_args'].get('global_epoch', 0)

        # 1) Prefer HPs prepared by the workflow for THIS epoch
        mailbox_hps = shared_state.get_and_pop_next_hps(client_id, this_epoch)
        if mailbox_hps is not None:
            hps = mailbox_hps
            current_state['concrete_hps'] = hps

        # 2) Else, reuse any concrete HPs we already have
        elif current_state.get('concrete_hps'):
            hps = current_state['concrete_hps']

        # 3) Else, first-ever time for this client: call "initial" ONCE
        else:
            # Guard against accidental double initial in the same epoch
            if not shared_state.mark_suggest_once(client_id, this_epoch, "initial"):
                print(f"[SKIP] Duplicate initial SUGGEST for client {client_id} epoch {this_epoch}")
                hps = current_state.get('concrete_hps', self._get_initial_hps(self.client_states[client_id]['search_space']))
            else:
                hps = self._get_reasoned_initial_hps(context)
                current_state['concrete_hps'] = hps



        # The rest of the function proceeds with a valid set of HPs
        client_hps = hps.get('client', {})
        batch_size = client_hps.get('batch_size', 32)
        dataset = context['train_subsets'][client_id]

        if len(dataset) <= 1:
            print(f"  - WARNING: Client {client_id} dataset is too small. Skipping.")
            return hps, None, None, 0, {"hps": hps, "results": {"error": "Dataset too small"}}

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        print(f"[GPU Worker]: Training client {client_id}...")
        training_args_with_loader = {**context['training_args'], 'train_loader': train_loader}

        training_start_time = time.time()

        w_c, w_s, sz, results = train_single_client(
            hps=hps,
            cid=client_id,
            **training_args_with_loader
        )

    

        training_end_time = time.time()
        client_training_time = training_end_time - training_start_time
        print(f"[GPU Worker]: Finished training client {client_id}. Total Time: {client_training_time:.2f}s")

        # --- Append & publish per-client history for reward/instability ---
        st = self.client_states[client_id]

        # last metrics
        try:
            acc = float(results.get('test_acc', results.get('acc', [0.0]))[-1])
        except Exception:
            acc = 0.0
        try:
            loss = float(results.get('test_loss', results.get('loss', [0.0]))[-1])
        except Exception:
            loss = 0.0

        # hp jump & mu
        st['prev_hps'] = st.get('hps_used', {})
        st['hps_used'] = hps
        try:
            st['mu_used'] = float(hps.get('mu', 0.0))
        except Exception:
            st['mu_used'] = 0.0

        # rolling windows
        W = getattr(self, 'history_window', 5)
        st.setdefault('recent_accs', []).append(acc)
        st.setdefault('recent_losses', []).append(loss)
        if len(st['recent_accs']) > W:   st['recent_accs']   = st['recent_accs'][-W:]
        if len(st['recent_losses']) > W: st['recent_losses'] = st['recent_losses'][-W:]

        # publish to shared_state so workflow can read immediately
        try:
            shared_state.CLIENT_METRICS  # ensure namespace exists
        except AttributeError:
            shared_state.CLIENT_METRICS = {}
        shared_state.CLIENT_METRICS[client_id] = {
            "recent_accs": st['recent_accs'],
            "recent_losses": st['recent_losses'],
            "hps_used": st['hps_used'],
            "prev_hps": st['prev_hps'],
            "mu_used": st['mu_used'],
        }



        # Offload the results to the background worker for the *next* round's suggestion
        results_for_analyzer = {
            "client_id": client_id, "results": results, "current_hps": hps,
            "cluster_id": context['cluster_id'], "model_name": context['model_name'],
            "dataset_name": context['dataset_name'], "global_epoch": context['training_args']['global_epoch'],
            "peer_history": context.get("peer_history"),
            "training_time": client_training_time ,
            "detailed_log_filename": self.detailed_log_filename
        }
        results_queue.put((client_id, results_for_analyzer))

        final_state = {
            "hps": hps, "results": results, "client_weights": w_c,
            "server_weights": w_s, "data_size": sz,
            "last_analysis": current_state.get('last_analysis', {})
        }

        return (
            final_state.get('hps'), final_state.get('client_weights'),
            final_state.get('server_weights'), final_state.get('data_size'),
            final_state
        )

class FixedStrategy(HPOStrategy):
    """
    A simple strategy for running baselines with fixed hyperparameters.
    """
    def __init__(self, initial_search_space: dict, client_states: list, **kwargs):
        super().__init__(initial_search_space, client_states, **kwargs)
        self.fixed_hps = kwargs.get('fixed_hps', {})
        if not self.fixed_hps:
            raise ValueError("FixedStrategy requires a 'fixed_hps' dictionary.")

    def get_hyperparameters(self, context: dict) -> tuple:
        from ssfl.trainer_utils import train_single_client
        print(f"--- Training Client {context['client_id']} with Fixed HPs: {self.fixed_hps} ---")
        w_c, w_s, sz, _ = train_single_client(**context['training_args'], hps=self.fixed_hps, cid=context['client_id'])
        # The 'final_state' is None for non-agent strategies, so no update will be called.
        return self.fixed_hps, w_c, w_s, sz, None


class RandomSearchStrategy(HPOStrategy):
    """
    An HPO strategy that suggests a new random set of hyperparameters for each client turn.
    """
    def _generate_random_hps(self) -> dict:
        """Helper to generate one random HP configuration."""
        random_hps = {}
        for name, config in self.initial_search_space.items():
            param_type = config.get('type')
            if param_type == 'float':
                random_hps[name] = random.uniform(config['min'], config['max'])
            elif param_type == 'int':
                random_hps[name] = random.randint(config['min'], config['max'])
            elif param_type == 'choice':
                random_hps[name] = random.choice(config['values'])
        return random_hps

    def get_hyperparameters(self, context: dict) -> tuple:
        from ssfl.trainer_utils import train_single_client
        random_hps = self._generate_random_hps()
        print(f"--- Training Client {context['client_id']} with Random HPs: {random_hps} ---")
        w_c, w_s, sz, results = train_single_client(**context['training_args'], hps=random_hps, cid=context['client_id'])
        return random_hps, w_c, w_s, sz, None

# NOTE: SHA and BO strategies remain unchanged as they do not use the agent workflow.
# I have omitted them here for brevity but you should keep them in your file.
class SHA_Strategy(HPOStrategy):
    """
    A functional implementation of Successive Halving (SHA).
    It manages a population of configurations and prunes the worst half at each "rung".
    """
    def __init__(self, initial_search_space: dict, client_states: list, **kwargs):
        super().__init__(initial_search_space, client_states, **kwargs)
        # --- SHA Specific State ---
        self.population_size = kwargs.get('population_size', 27) # Total configs to start with
        self.elimination_rate = kwargs.get('elimination_rate', 3) # e.g., eliminate all but 1/3
        self.rung = 0 # Current stage of elimination
        self.rung_evals = 0 # Number of evaluations in the current rung
        
        # Create the initial population of hyperparameter configurations
        self.population = [self._generate_random_hps() for _ in range(self.population_size)]
        self.performance = {i: [] for i in range(self.population_size)} # Store performance for each config

    def _generate_random_hps(self) -> dict:
        # (Same as in RandomSearchStrategy)
        random_hps = {}
        for name, config in self.initial_search_space.items():
            param_type = config.get('type')
            if param_type == 'float':
                random_hps[name] = random.uniform(config['min'], config['max'])
            elif param_type == 'choice':
                random_hps[name] = random.choice(config['values'])
        return random_hps

    def get_hyperparameters(self, context: dict) -> tuple:
        from ssfl.trainer_utils import train_single_client
        
        # Determine how many configs should be active in the current rung
        num_active_configs = self.population_size // (self.elimination_rate ** self.rung)
        
        # Pick the next configuration to evaluate
        config_index = self.rung_evals % num_active_configs
        hps_to_use = self.population[config_index]

        print(f"--- SHA Rung {self.rung}: Evaluating config {config_index+1}/{num_active_configs} for Client {context['client_id']} ---")
        
        w_c, w_s, sz, results = train_single_client(**context['training_args'], hps=hps_to_use, cid=context['client_id'])
        
        # Record performance
        self.performance[config_index].append(results['test_acc'][-1])
        self.rung_evals += 1
        
        # Check if it's time to prune the population and advance to the next rung
        # A simple check: if we've evaluated each active config once
        if self.rung_evals >= num_active_configs:
            print(f"--- SHA Rung {self.rung} complete. Pruning population... ---")
            
            # Calculate average performance for each config in this rung
            avg_performance = {i: np.mean(self.performance[i]) for i in range(num_active_configs)}
            
            # Sort configs by performance (higher is better)
            sorted_configs = sorted(avg_performance.items(), key=lambda item: item[1], reverse=True)
            
            # Keep the top 1/N fraction
            num_to_keep = num_active_configs // self.elimination_rate
            
            # Get the indices of the configurations to keep
            top_indices = [item[0] for item in sorted_configs[:num_to_keep]]
            
            # Create the new population for the next rung
            self.population = [self.population[i] for i in top_indices]
            self.performance = {i: self.performance[top_indices[i]] for i in range(len(self.population))} # Reset keys
            
            self.rung += 1
            self.rung_evals = 0 # Reset for the new, smaller rung
            print(f"--- Advanced to SHA Rung {self.rung}. New population size: {len(self.population)} ---")

        return hps_to_use, w_c, w_s, sz, None


class BO_Strategy(HPOStrategy):
    """
    A structural placeholder for a Bayesian Optimization (BO) strategy.
    This shows how to integrate a library like 'scikit-optimize'.
    """
    def __init__(self, initial_search_space: dict, client_states: list, **kwargs):
        super().__init__(initial_search_space, client_states, **kwargs)
        try:
            from skopt import Optimizer
            from skopt.space import Real, Categorical
        except ImportError:
            raise ImportError("To use BO_Strategy, please install scikit-optimize: `pip install scikit-optimize`")
        
        self.num_clients = kwargs.get('num_clients',1)

        # --- Setup for Bayesian Optimization ---
        # 1. Define the search space in the format the library expects
        self.bo_space = []
        self.hp_names = []
        for name, config in self.initial_search_space.items():
            self.hp_names.append(name)
            if config['type'] == 'float':
                self.bo_space.append(Real(config['min'], config['max'], name=name))
            elif config['type'] == 'choice':
                self.bo_space.append(Categorical(config['values'], name=name))
        
        # 2. Create a separate optimizer for each client to track their individual performance
        self.client_optimizers = {i: Optimizer(self.bo_space) for i in range(self.num_clients)}

    def get_hyperparameters(self, context: dict) -> tuple:
        from ssfl.trainer_utils import train_single_client
        client_id = context['client_id']
        optimizer = self.client_optimizers[client_id]

        # 1. ASK the optimizer for the next best hyperparameters to try
        suggested_params_list = optimizer.ask()
        hps = {name: value for name, value in zip(self.hp_names, suggested_params_list)}
        
        print(f"--- BO requesting Client {client_id} to test HPs: {hps} ---")

        # 2. Train the model with these HPs
        w_c, w_s, sz, results = train_single_client(**context['training_args'], hps=hps, cid=client_id)

        # 3. TELL the optimizer the result (we want to maximize accuracy, so we minimize the negative)
        # The score should be a single float. We'll use the final test accuracy.
        score = -results['test_acc'][-1] 
        optimizer.tell(suggested_params_list, score)
        
        return hps, w_c, w_s, sz, None