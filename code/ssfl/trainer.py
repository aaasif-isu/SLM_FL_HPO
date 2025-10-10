# # code/ssfl/trainer.py

import torch
import yaml
import os
from torch.utils.data import ConcatDataset, DataLoader

# Original imports
from ssfl.model_splitter import create_global_model
from ssfl.utils import ensure_dir, save_model
from ssfl.trainer_utils import (
    prepare_training, select_participating_clients, build_cluster_model, 
    evaluate_model, _format_report, train_single_client, log_global_metrics
)
from ssfl.trainer_utils import cluster_fedavg, global_fedavg

# --- 1. Import the new strategy classes ---
# Make sure your strategies.py file is in the ssfl folder or adjust path.
from ssfl.strategies import AgentStrategy, FixedStrategy, RandomSearchStrategy, BO_Strategy , SHA_Strategy


def train_model(model_name, num_classes, in_channels,
                train_subsets, val_loader,
                device, global_epochs,
                num_clients, imbalance_ratio, dataset_name, frac,
                config, client_states): # Pass the whole config dictionary

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    global_model = create_global_model(model_name, num_classes, in_channels, device)

    # --- Load HPO Search Space ---
    hp_config_path = os.path.join(os.path.dirname(__file__), "..", "agent", "hp_config.yaml")
    with open(hp_config_path, 'r') as f:
        initial_search_space = yaml.safe_load(f)

    # --- HPO STRATEGY SELECTION ---
    hpo_config = config.get('hpo_strategy', {})
    history_window = hpo_config.get('history_window', 5)
    strategy_name = hpo_config.get('method', 'fixed') # Default to 'fixed' if not specified

    # --- Get FL Mode ---
    fl_mode = config.get('fl_mode', 'splitfed') # Default to 'splitfed'
    print(f"--- Using FL Mode: {fl_mode.upper()} ---")

    # Commented out for the decoupling CPU GPU task 
    #client_states = [{"search_space": initial_search_space.copy(), "hpo_report": {}, "last_analysis": None} for i in range(num_clients)]

    hpo_strategy = None

    fl_mode = config.get('fl_mode', 'centralized')

    detailed_log_filename = (
        f"{model_name}__{dataset_name}_clients{num_clients}_"
        f"imb{imbalance_ratio}_epochs{global_epochs}_{fl_mode}_training_details_metrics.csv"
    )

    # Common arguments for all strategies
    strategy_args = {
        "initial_search_space": initial_search_space,
        "client_states": client_states,
        "num_clients": num_clients,
        "history_window": history_window,
        "detailed_log_filename": detailed_log_filename
    }

    if strategy_name == 'agent':
        hpo_strategy = AgentStrategy(**strategy_args)
    elif strategy_name == 'random_search':
        hpo_strategy = RandomSearchStrategy(**strategy_args)
    elif strategy_name == 'sha':
        sha_config = hpo_config.get('sha_config', {})
        hpo_strategy = SHA_Strategy(**strategy_args, **sha_config)
    elif strategy_name == 'bo':
        hpo_strategy = BO_Strategy(**strategy_args)
    else:
        strategy_args['fixed_hps'] = hpo_config.get('fixed_hps')
        hpo_strategy = FixedStrategy(**strategy_args)
    print(f"--- Using HPO Strategy: {strategy_name.upper()} ---")

    # --- Load Training Control Params (No Hardcoding) ---
    training_params = config.get('training_params', {})
    patience = training_params.get('patience', 10)
    min_delta = training_params.get('min_delta', 0.01)

    # --- Setup code from original trainer (Unchanged) ---
    best_global_accuracy, no_improvement_count = 0.0, 0
    best_model_path = f"best_model/best_{dataset_name}_c{num_clients}_imb{imbalance_ratio}.pth"
    ensure_dir("best_model")

    # Pass fl_mode to prepare_training
    arc_configs, clients_per_cluster, total_layers  = prepare_training(
        model_name, global_model, num_clients, config.get('num_clusters', 3), fl_mode
    )

    # --- ADD THIS DEBUG PRINT ---
    # print(f"DEBUG: Type of clients_per_cluster: {type(clients_per_cluster)}")
    # print(f"DEBUG: Value of clients_per_cluster: {clients_per_cluster}")


    # --- Main Training Loop ---
    for g_epoch in range(global_epochs):
        print(f"\n=== Global Epoch {g_epoch+1}/{global_epochs} ===")

        # [ADD] round metadata
        round_idx = g_epoch + 1
        total_rounds = global_epochs
        phase = "train" if round_idx < total_rounds else "finalize"

        selected = select_participating_clients(num_clients, frac)

        # Initialize lists for aggregation based on FL mode
        cluster_state_dicts = [] # Stores state_dicts of cluster models for final global aggregation
        cluster_sizes = [] # Stores sum of client sizes for each cluster for final global aggregation

        # For centralized, we will collect their full model updates here for direct global aggregation.
        local_full_model_updates = []
        # local_sizes will be used for both centralized and splitfed client sizes

        # --- MODIFIED LOGIC FOR UNIQUE_CLUSTER_IDS ---
        # This makes the code resilient to clients_per_cluster being either a dict or a list.
        if isinstance(clients_per_cluster, dict):
            unique_cluster_ids = sorted(list(set(clients_per_cluster.values())))
        elif isinstance(clients_per_cluster, list):
            unique_cluster_ids = sorted(list(set(clients_per_cluster)))
        else:
            raise TypeError(f"clients_per_cluster has unexpected type: {type(clients_per_cluster)}")
        # --- END MODIFIED LOGIC ---

        for c_id in unique_cluster_ids:
            # For centralized, arc_cfg will be effectively ignored in train_single_client logic
            arc_cfg = arc_configs[c_id] if fl_mode == 'splitfed' and c_id < len(arc_configs) else 0

            # Determine members for the current cluster based on the actual type of clients_per_cluster
            if isinstance(clients_per_cluster, dict):
                # If clients_per_cluster is a dict, use .get()
                members = [cid for cid in selected if clients_per_cluster.get(cid, 0) == c_id]
            elif isinstance(clients_per_cluster, list):
                # If clients_per_cluster is a list, assume it's a direct mapping (client_id -> cluster_id)
                # Ensure cid is a valid index for the list before accessing
                members = [cid for cid in selected if cid < len(clients_per_cluster) and clients_per_cluster[cid] == c_id]
            else:
                raise TypeError(f"clients_per_cluster has unexpected type: {type(clients_per_cluster)}")


            if not members:
                continue

            print(f"\n***Cluster {c_id} (FL Mode: {fl_mode.upper()}) with members {members}***")

            local_client_w, local_server_w, local_sizes = [], [], [] # Will be used differently based on fl_mode
            cluster_peer_history = []

            for cid in members:
                # --- DELEGATE TO THE STRATEGY ---
                context = {
                    "client_id": cid,
                    "cluster_id": c_id,
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "peer_history": cluster_peer_history, # Pass the history of previous peers
                    "arc_cfg": arc_cfg, # Still passed, but ignored in centralized mode within train_single_client
                    "total_layers": total_layers,
                    "train_subsets": train_subsets,
                    "global_epochs": global_epochs,

                    # [ADD] round metadata for downstream (AgentStrategy → queue → CPU worker → graph)
                    "round_idx": round_idx,
                    "total_rounds": total_rounds,
                    "phase": phase,


                    "training_args": {
                        "model_name": model_name, "num_classes": num_classes, "arc_cfg": arc_cfg,
                        "global_model": global_model, "device": device, "in_channels": in_channels,
                        "val_loader": val_loader, "loss_fn": loss_fn,
                        "global_epoch": g_epoch,
                        "fl_mode": fl_mode, # Pass fl_mode to training args



                    }
                }

                # The strategy object handles all HPO and training logic
                # w_c will be client_net.state_dict() for splitfed, or full_model.state_dict() for centralized
                # w_s will be server_net.state_dict() for splitfed, or None for centralized
                hps, w_c, w_s, sz, final_state = hpo_strategy.get_hyperparameters(context)

                # Append results for aggregation based on FL mode
                if fl_mode == 'splitfed':
                    local_client_w.append(w_c)
                    local_server_w.append(w_s)
                elif fl_mode == 'centralized':
                    local_full_model_updates.append(w_c) # w_c now holds the full model state dict
                local_sizes.append(sz) # This is always valid (dataset size)

                # Call the updated state method, now passing the global epoch
                if final_state is not None and isinstance(hpo_strategy, AgentStrategy):
                    hpo_strategy.update_persistent_state(cid, context, final_state)

                    # --- NEW: Create a summary of the completed run for the next peer ---
                    if final_state.get('last_analysis'):
                        key_insight = final_state.get('last_analysis', {}).get('decision_summary', 'Analysis failed.')
                        # hps_summary = final_state.get('hps', {}) # Already in final_state
                        # results_summary = final_state.get('results', {}) # Already in final_state
                        test_acc_summary = final_state.get('results', {}).get('test_acc', [0.0])[-1]

                        peer_summary = {
                            "client_id": cid,
                            "hps_used": final_state.get('hps', {}),
                            "result_and_decision": f"Achieved {test_acc_summary:.2f}% Acc. Analyzer Decision: '{key_insight}'"
                        }
                        cluster_peer_history.append(peer_summary)


            # --- Aggregation and Evaluation Logic (Adjusted for FL Mode) ---
            if fl_mode == 'splitfed':
                if not local_client_w: # if no clients participated in this cluster
                    continue

                w_c_agg, w_s_agg = cluster_fedavg(local_client_w, local_server_w, local_sizes)
                cluster_model = build_cluster_model(model_name, num_classes, arc_cfg, global_model, device, in_channels, w_c_agg, w_s_agg)

                cluster_train_subset = ConcatDataset([train_subsets[i] for i in members])
                cluster_train_loader = DataLoader(cluster_train_subset, batch_size=128, shuffle=True, drop_last=True)
                acc_train, _ = evaluate_model(cluster_model, cluster_train_loader, device, loss_fn)
                acc_test, _ = evaluate_model(cluster_model, val_loader, device, loss_fn)
                print(f"  Cluster {c_id} Train Acc {acc_train:.2f}%, Test Acc {acc_test:.2f}%")

                cluster_state_dicts.append(cluster_model.state_dict())
                cluster_sizes.append(sum(local_sizes))

            elif fl_mode == 'centralized':
                if not local_full_model_updates: # if no clients participated in this logical cluster
                    continue

                # Aggregate full model updates directly into global_model
                # Use global_fedavg for this, which correctly averages state_dicts
                aggregated_full_model_state_dict = global_fedavg(local_full_model_updates, local_sizes)
                global_model.load_state_dict(aggregated_full_model_state_dict)

                # For centralized, we append the current global model state to cluster_state_dicts
                # This ensures the final global_fedavg outside the cluster loop still works
                cluster_state_dicts.append(global_model.state_dict())
                cluster_sizes.append(sum(local_sizes)) # Sum of sizes from all clients in this global epoch

                # Evaluate the current global model state for printout in centralized mode
                all_train_subset_for_eval = ConcatDataset([train_subsets[i] for i in members]) # Evaluate on participating clients' data
                all_train_loader_for_eval = DataLoader(all_train_subset_for_eval, batch_size=128, shuffle=False, drop_last=True)
                g_train_acc, _ = evaluate_model(global_model, all_train_loader_for_eval, device, loss_fn)
                g_test_acc, _ = evaluate_model(global_model, val_loader, device, loss_fn)
                print(f"  Global Model Update (Centralized): Train Acc {g_train_acc:.2f}%, Test Acc {g_test_acc:.2f}%")


        if cluster_state_dicts: # This will be true for both modes
            global_weights = global_fedavg(cluster_state_dicts, cluster_sizes)
            global_model.load_state_dict(global_weights)

            all_train_subset = ConcatDataset(train_subsets)
            all_train_loader = DataLoader(all_train_subset, batch_size=128, shuffle=False, drop_last=True)
            g_train_acc, g_train_loss = evaluate_model(global_model, all_train_loader, device, loss_fn)
            g_test_acc, g_test_loss = evaluate_model(global_model, val_loader, device, loss_fn)
            print(f"Global Epoch {g_epoch+1}: Train Acc {g_train_acc:.2f}%, Test Acc {g_test_acc:.2f}%")

            log_global_metrics(
                model_name=model_name,
                dataset_name=dataset_name,
                epoch=g_epoch + 1,
                metrics={
                    'global_train_acc': round(g_train_acc, 4),
                    'global_train_loss': round(g_train_loss, 4),
                    'global_test_acc': round(g_test_acc, 4),
                    'global_test_loss': round(g_test_loss, 4)
                },
                num_clients=num_clients,
                imbalance_ratio=imbalance_ratio,
                global_epochs=global_epochs,
                fl_mode=fl_mode
            )

            if g_test_acc > best_global_accuracy + min_delta:
                best_global_accuracy, no_improvement_count = g_test_acc, 0
                save_model(global_model, best_model_path)
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print("Early stopping.")
                    break

    print(f"Best Global Accuracy: {best_global_accuracy:.2f}%")

    # # --- Final Save of HPO States ---
    # # Get the relative path from your config file
    # relative_path = config.get('hpo_checkpoint_path', 'agent/client_hpo_states.yaml')

    # # --- THIS IS THE SIMPLIFIED FIX ---
    # # Construct the correct path by going up one directory ('..') from the current script's location
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # final_states_path = os.path.join(current_dir, '..', relative_path)
    
    # print(f"\n--- Saving final HPO states for all clients to {final_states_path} ---")

    try:

        # 1. Get the base directory from your config file (e.g., 'agent/').
        hpo_config = config.get('hpo_strategy', {})
        relative_path = hpo_config.get('hpo_checkpoint_path', 'agent/client_hpo_states.yaml')
        output_folder_name = os.path.dirname(relative_path)

        fl_mode = config.get('fl_mode', 'splitfed')

        # ===================================================================
        # --- 2. CREATE A UNIQUE FILENAME WITH ALL KEY PARAMETERS ---
        # ===================================================================
        # This now includes the imbalance ratio and FL mode to prevent overwrites.
        # Example: hpo_state_ResNet18__pacs_imb0.1_splitfed.yaml
        experiment_specific_filename = (
            f"hpo_state_{model_name}__{dataset_name}_clients{num_clients}_"
            f"imb{imbalance_ratio}_epochs{global_epochs}_{fl_mode}.yaml"
        )
        # ===================================================================

        # 3. Construct the full, absolute path from the project root.
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_script_dir)
        final_states_path = os.path.join(project_root, output_folder_name, experiment_specific_filename)

        
        print(f"\n--- Saving final HPO states for this experiment to: {final_states_path} ---")

        # 4. Ensure the directory exists before saving.
        #os.makedirs(os.path.dirname(final_save_path), exist_ok=True)




        final_states_to_save = _format_report(hpo_strategy.client_states)
        

        # Add the client_id to each state for clarity before saving
        for i, state in enumerate(final_states_to_save):
            state['client_id'] = i
            # Rename search_space to final_search_space for clarity in the final report
            if 'search_space' in state:
                state['final_search_space'] = state.pop('search_space')


        checkpoint_dir = os.path.dirname(final_states_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            
        with open(final_states_path, 'w') as f:
            import copy
            final_report_to_dump = copy.deepcopy(final_states_to_save)

            yaml.dump(final_report_to_dump, f, indent=4, default_flow_style=False, sort_keys=False)

            # yaml.dump(final_states_to_save, f, indent=4, sort_keys=False, default_flow_style=False)
        print("--- Final states saved successfully. ---")

    except Exception as e:
        print(f"Error: Could not save final HPO states. {e}")

    return global_model