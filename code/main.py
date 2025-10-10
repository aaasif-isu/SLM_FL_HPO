# code/main.py
import yaml
import torch
import os
import sys
import logging
import threading # <-- 1. IMPORT THREADING
import argparse

# --- Your existing imports ---
from ssfl.utils import (
    load_dataset,
    partition_data_non_iid_random,
    partition_text_non_iid_dirichlet,
    subsample_dataset,
    Tee
)
from ssfl.trainer import train_model

# --- Imports for the new parallel workflow ---
from agent.cpu_worker import background_cpu_work # <-- 2. IMPORT THE CPU WORKER
from agent.shared_state import results_queue     # <-- 3. IMPORT THE SHARED QUEUE

import json
#from agent.shared_state import HP_AGENT_STATS, ANALYZER_AGENT_STATS, load_stats
from agent.shared_state import aggregate_hp_events, aggregate_analyzer_events
from agent.shared_state import reset_aggregates
reset_aggregates()

def load_config(path="model_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run a federated learning experiment.")
    parser.add_argument(
        '--config',
        type=str,
        default="model_config.yaml",
        help='Path to the configuration file for the experiment.'
    )
    args = parser.parse_args()

    config = load_config(args.config)


    #config = load_config()

    # --- Logging Setup (remains the same) ---
    log_dir = config.get("log_dir", "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"{config.get('num_clients', 'N')}_clients_run.log")
    try:
        sys.stdout = Tee(log_filename)
    except Exception as e:
        print(f"Failed to redirect stdout: {e}")

    # --- Data Loading and Partitioning (remains the same) ---
    train_dataset, test_dataset, num_classes, image_size, in_channels = load_dataset(config["dataset_name"])
    print(f"Dataset: {config['dataset_name'].upper()}")
    train_subset = subsample_dataset(train_dataset, config["train_sample_fraction"])
    test_subset = subsample_dataset(test_dataset, config["test_sample_fraction"])

    if config["dataset_name"].lower() == 'shakespeare_d':
        # This assumes a new function, 'get_data_by_client', which loads the
        # naturally partitioned data from all_data.json
        all_client_data = get_data_by_client(dataset_path=config["data_path"], dataset_name='shakespeare')
        
        train_subsets = []
        # The number of clients is now determined by the data itself
        num_clients = len(all_client_data.keys()) 

        for client_id in all_client_data.keys():
            # You would need to create a Dataset or Subset from the client's data
            # 'all_client_data[client_id]' contains the grouped data for one actor
            client_dataset = create_subset_from_grouped_data(all_client_data[client_id])
            train_subsets.append(client_dataset)

        
        print(f"Loaded {num_clients} clients from the dataset's inherent partitioning.")

    elif config["dataset_name"].lower() == "officehome":
        # If we got a list of datasets, use them directly as the client subsets
        print("--- Office-Home detected. Assigning one domain per client. ---")
        train_subsets = train_dataset 
        
        # Optional: Check that the config matches the data
        if len(train_subsets) != config["num_clients"]:
            print(f"Warning: Config specifies {config['num_clients']} clients, but {len(train_subsets)} domains were loaded.")

    elif config["dataset_name"].lower() == "pacs":
        print("--- PACS detected. Assigning one domain per client. ---")
        # Logic is identical to OfficeHome
        train_subsets = train_dataset
        if len(train_subsets) != config["num_clients"]:
            print(f"Warning: Config specifies {config['num_clients']} clients, but {len(train_subsets)} domains were loaded.")



    elif config["dataset_name"].lower() == 'shakespeare':
        train_subsets = partition_text_non_iid_dirichlet(
            dataset=train_subset,
            num_clients=config["num_clients"],
            imbalance_factor=config["imbalance_ratio"], # You can reuse this config
            min_samples_per_client=config["min_samples_per_client"] 
        )
    else:
        train_subsets = partition_data_non_iid_random(
            train_subset, config["num_clients"], config["imbalance_ratio"], config["min_samples_per_client"]
            )

    # --- DEBUGGING BLOCK  ---
    # print("\n" + "="*40)
    # print("--- Inspecting Client Data ---")
    
    # # Check the total number of subsets created
    # print(f"Total client subsets created: {len(train_subsets)}")

    # # Loop through all clients to print their ID and dataset size
    # for client_id, client_subset in enumerate(train_subsets):
    #     print(f"Client {client_id} has {len(client_subset)} samples.")

       

    # print("="*40 + "\n")
    # --- END DEBUGGING BLOCK ---

    val_batch_size = config.get('val_batch_size', 128)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=val_batch_size, shuffle=False, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 4. PREPARE SHARED STATE FOR PARALLEL WORK ---
    # This logic is moved here from trainer.py to be shared between threads.
    hp_config_path = os.path.join("agent", "hp_config.yaml")
    with open(hp_config_path, 'r') as f:
        initial_search_space = yaml.safe_load(f)

    num_clients = config['num_clients']
    client_states = [{"search_space": initial_search_space.copy(), "concrete_hps": None, "hpo_report": {}, "last_analysis": None} for _ in range(num_clients)]

    # --- 5. START BACKGROUND CPU WORKER ---
    print("Starting background CPU worker...")
    # NOTE: The agents are instantiated inside your workflow, so we only pass client_states.
    worker_thread = threading.Thread(
        target=background_cpu_work,
        args=(client_states,)
    )
    worker_thread.daemon = False # True means :This allows the main program to exit even if the thread is running.
    worker_thread.start()

    print(f"Training with {num_clients} clients.")
    print(f"Non-IIDness: {config['imbalance_ratio']}, Dataset: {config['dataset_name']}")

    # --- 6. MODIFIED CALL TO train_model ---
    # This now includes the client_states argument at the end.
    train_model(
        model_name=config["model_name"],
        num_classes=num_classes,
        in_channels=in_channels,
        train_subsets=train_subsets,
        val_loader=test_loader,
        device=device,
        global_epochs=config["global_epochs"],
        num_clients=num_clients,
        imbalance_ratio=config["imbalance_ratio"],
        dataset_name=config["dataset_name"],
        frac=config["frac"],
        config=config,
        client_states=client_states # <-- Pass the shared state object
    )

    # --- 7.  SHUTDOWN ---
    print("\nEpochs finished. Finalizing analysis…")
    results_queue.join()                # wait until the worker processed every queued item
    results_queue.put((None, None))     # now send sentinel to stop the worker loop
    worker_thread.join()                # cleanly wait for the worker to exit
    print(" All tasks complete. System shutting down.")

    print("\n===== HP Agent Summary (Aggregated) =====")
    print(json.dumps(aggregate_hp_events(), indent=2))
    print("===== Analyzer Agent Summary (Aggregated) =====")
    print(json.dumps(aggregate_analyzer_events(), indent=2))
    print("==========================================\n")



    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print("Training completed.")

if __name__ == "__main__":
    main()

