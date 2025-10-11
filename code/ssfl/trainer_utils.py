# code/ssfl/trainer_utils.py
import torch
from torch.nn.utils import clip_grad_norm_
from ssfl.aggregation import FedAvg, combine_client_server_models
from ssfl.model_splitter import create_split_model, get_total_layers, create_global_model
from ssfl.resource_profiler import profile_resources
from ssfl.utils import calculate_accuracy, save_model
from ssfl.utils_seed import seed_everything
import numpy as np
import random
from torch.utils.data import ConcatDataset, DataLoader
import copy

import os
import csv
import threading
from typing import Dict

# This global lock ensures that only one thread can write to the CSV at a time.
CSV_LOCK = threading.Lock()

def log_epoch_metrics(
    detailed_log_filename: str,
    model_name: str,
    dataset_name: str,
    epoch: int,
    client_id: int,
    metrics: Dict
):
    """
    Logs performance metrics for a single client to a model-specific CSV file.
    This function is designed to be thread-safe.

    Args:
        model_name (str): The name of the model, used for the CSV filename.
        epoch (int): The current global epoch.
        client_id (int): The ID of the client.
        metrics (Dict): A dictionary of metrics to log (e.g., {'training_time': 10.5, 'llm_latency': 2.1}).
    """
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # The filename is generated dynamically from the model config to avoid hardcoding.
    #log_file = os.path.join(results_dir, f"{model_name}__{dataset_name}_training_metrics.csv")
    log_file = os.path.join(results_dir, detailed_log_filename)


    # The 'with' statement acquires the lock before entering the block
    # and automatically releases it upon exiting, even if errors occur.
    with CSV_LOCK:
        file_exists = os.path.isfile(log_file)
        
        # We use DictWriter for robustness; it handles the columns by name.
        fieldnames = ['epoch', 'client_id'] + list(metrics.keys())
        
        with open(log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write the header row only if the file is brand new.
            if not file_exists:
                writer.writeheader()
            
            # Combine the standard info with the metrics and write the row.
            row_to_write = {'epoch': epoch, 'client_id': client_id, **metrics}
            writer.writerow(row_to_write)



def create_optimizer(model_params, hps: dict):
    """Dynamically creates an optimizer based on hyperparameter suggestions."""
    optimizer_name = hps.get('optimizer', 'AdamW')
    lr = hps.get('learning_rate', 0.001)
    wd = hps.get('weight_decay', 0)
    momentum = hps.get('momentum', 0.9)

    if optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(model_params, lr=lr, weight_decay=wd, momentum=momentum)
    # Add other optimizers here if needed
    else:  # Default to AdamW
        return torch.optim.AdamW(model_params, lr=lr, weight_decay=wd)

def create_scheduler(optimizer, hps: dict, T_max: int):
    """Dynamically creates a learning rate scheduler."""
    scheduler_name = hps.get('scheduler', 'None')
    
    if scheduler_name.lower() == 'cosineannealinglr':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_name.lower() == 'steplr':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, T_max // 3), gamma=0.1)
    else:
        return None

def prepare_training(model_name, global_model, num_clients, num_clusters=3, fl_mode="splitfed"):
    """
    Prepares training configurations based on the FL mode.
    For 'splitfed', it sets up arc_configs and client-to-cluster mappings.
    For 'centralized', it simplifies to a single logical cluster.
    """
    if fl_mode == "splitfed":
        total_layer = get_total_layers(global_model)
        print(f"\nTotal layer in {model_name} is {total_layer}")
        # Ensure arc_configs are valid; total_layer - 1 is max split point for non-empty server
        #arc_configs = np.linspace(1, max(1, total_layer - 1), num_clusters, dtype=int).tolist()
        # --- START OF THE FIX ---
    
        # Check if the model is BERT and apply its specific, valid split configurations
        if 'bert' in model_name.lower():
            # These are the only valid split points for the create_split_bert function
            valid_bert_configs = [0, 6, 9, 12]
            
            if num_clusters > len(valid_bert_configs):
                print(f"Warning: num_clusters ({num_clusters}) is > number of valid BERT splits (4). Using all {len(valid_bert_configs)} valid splits.")
                arc_configs = valid_bert_configs
            else:
                # This selects the last 'num_clusters' from the valid list, which prioritizes deeper splits.
                # e.g., if num_clusters=1, it uses [12]. if num_clusters=3, it uses [6, 9, 12].
                arc_configs = valid_bert_configs[-num_clusters:]
                
            print(f"Using model-specific BERT arc_configs: {arc_configs}")
        
        else:
            # For any other model (ResNet, CNN, etc.), use the generic linspace approach
            print("Using generic linspace for arc_configs.")
            arc_configs = np.linspace(1, max(1, total_layer - 1), num_clusters, dtype=int).tolist()

        # --- END OF THE FIX ---

        clients_per_cluster = profile_resources(num_clients, num_clusters)
        return arc_configs, clients_per_cluster, total_layer
    elif fl_mode == "centralized":
        print("\n--- Centralized FL Mode: No model splitting for local training ---")
        # For centralized, we treat all clients as part of a single logical cluster.
        # arc_configs can be a dummy list, as it won't be used for splitting in train_single_client.
        arc_configs = [0] # A single dummy arc_config
        # All clients belong to cluster 0
        clients_per_cluster = {i: 0 for i in range(num_clients)}
        total_layer = get_total_layers(global_model) # Still relevant for model info
        return arc_configs, clients_per_cluster, total_layer
    else:
        raise ValueError(f"Unknown FL mode: {fl_mode}. Supported modes are 'splitfed' and 'centralized'.")


def prepare_training_old(model_name, global_model, num_clients, num_clusters=3):
    total_layer = get_total_layers(global_model)
    print(f"\nTotal layer in {model_name} is {total_layer}")
    arc_configs = np.linspace(1, total_layer - 1, num_clusters, dtype=int).tolist()
    clients_per_cluster = profile_resources(num_clients, num_clusters)
    return arc_configs, clients_per_cluster, total_layer


def select_participating_clients(num_clients, frac):
    k = max(1, int(num_clients * frac))
    return random.sample(range(num_clients), k)



def train_single_client(model_name, num_classes, arc_cfg, global_model,
                        device, in_channels, train_loader, val_loader, loss_fn,
                        cid, hps: dict, global_epoch: int, fl_mode: str): # Added fl_mode
    """
    Trains a single client model. Adapts based on 'splitfed' or 'centralized' mode.
    Handles both image (tuple) and text (dict) data formats.
    """
    seed_everything(seed=cid)
    results = {'train_loss': [],'test_loss': [], 'test_acc': [], 'train_acc': []}

    if fl_mode == "splitfed":
        client_net, server_net, full_ref, _ = create_split_model(
            model_name, num_classes, arc_cfg,
            base_model=global_model, device=device, in_channels=in_channels
        )

        client_hps = hps.get('client', {})
        server_hps = hps.get('server', {})
        mu = hps.get('mu', 0.01)
        local_epochs = int(client_hps.get('local_epochs', 1))

        opt_c = create_optimizer(client_net.parameters(), client_hps)
        opt_s = create_optimizer(server_net.parameters(), server_hps)

        sch_c = create_scheduler(opt_c, client_hps, T_max=local_epochs)
        sch_s = create_scheduler(opt_s, server_hps, T_max=local_epochs)

        global_client_params = [param.detach().clone() for param in client_net.parameters()]

        client_net.train()
        server_net.train()

        total_train_loss = 0.0
        num_train_batches = 0

        for epoch in range(local_epochs):
            for batch in train_loader:
                opt_c.zero_grad()
                opt_s.zero_grad()

                # =================== START OF NLP FIX (SPLITFED) ===================
                if isinstance(batch, dict):
                    # --- BRANCH 1: HANDLE TEXT DATA (BERT) ---
                    # 1. Move all tensors from the batch dictionary to the device
                    model_inputs = {k: v.to(device) for k, v in batch.items()}
                    
                    # 2. 'labels' are for the final loss, not a direct model input
                    lbls = model_inputs.pop("labels")
                    
                    # 3. Call client model with input_ids and attention_mask
                    client_feat = client_net(**model_inputs)
                    
                    # 4. Detach the output for the server forward pass
                    smashed = client_feat.detach().requires_grad_(True)
                    
                    # 5. Call server model with smashed data and the original attention mask
                    #    (This requires the BERTServer.forward to accept attention_mask)
                    out = server_net(smashed, model_inputs["attention_mask"])

                else:
                    # --- BRANCH 2: HANDLE IMAGE DATA (Your Original Logic) ---
                    imgs, lbls = batch
                    if imgs.shape[0] <= 1:
                        print(f"  --> WARNING: Skipping batch of size {imgs.shape[0]} for Client {cid} to prevent BatchNorm error.")
                        continue
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    
                    client_feat = client_net(imgs)
                    smashed = client_feat.detach().requires_grad_(True)
                    
                    #### The original server_net for images only needs the smashed data, but we need to think about passing the lbls
                    out = server_net(smashed) 
                
                # =================== END OF NLP FIX (SPLITFED) =====================

                #loss = loss_fn(out, lbls)

                # Before calculating loss, we must ensure tensors have the correct shape.
                if model_name.lower() in ['charlstm', 'bert']:
                    # For text models, output is [batch*seq_len, vocab_size]
                    # and labels are [batch, seq_len]. We need to flatten labels.
                    loss = loss_fn(out, lbls.view(-1))
                else:
                    # For image models, shapes are already correct.
                    loss = loss_fn(out, lbls)



                if mu > 0:
                    prox_term = 0.0
                    for local_param, global_param in zip(client_net.parameters(), global_client_params):
                        prox_term += torch.pow(torch.norm(local_param - global_param), 2)
                    loss += (mu / 2) * prox_term

                 # --- MODIFICATION 3: Accumulate training loss ---
                total_train_loss += loss.item()
                num_train_batches += 1

                loss.backward()
                # For split learning, the gradient from the server's input must be propagated back
                if isinstance(batch, tuple): # If it's an image model
                    client_feat.backward(smashed.grad)

                clip_grad_norm_(client_net.parameters(), 1.0)
                clip_grad_norm_(server_net.parameters(), 1.0)
                opt_c.step()
                opt_s.step()

                if sch_c: sch_c.step()
                if sch_s: sch_s.step()

            #❗❗❗  IMPORTANT: Maybe add this here!!later need to analyze this
            # if sch_c: sch_c.step()
            # if sch_s: sch_s.step()

        # Evaluation should also handle both data types, if needed.
        # Assuming evaluate_model is compatible or we primarily evaluate on centralized logic.
        temp_model = combine_client_server_models(client_net, server_net, full_ref.to(device), device, num_classes, arc_cfg)
        train_acc, _ = evaluate_model(temp_model, train_loader, device, loss_fn, model_name)
        test_acc, test_loss = evaluate_model(temp_model, val_loader, device, loss_fn, model_name)

        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0


        # results['train_acc'].append(train_acc)
        # results['test_acc'].append(test_acc)

        results['train_loss'].append(avg_train_loss)
        results['test_loss'].append(test_loss)
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)


        print(f"  Client {cid}, Local Epochs {local_epochs}: Train Acc {train_acc:.2f}%, Test Acc {test_acc:.2f}%")

        return client_net.state_dict(), server_net.state_dict(), len(train_loader.dataset), results

    elif fl_mode == "centralized":
        local_model = copy.deepcopy(global_model).to(device)
        client_hps = hps.get('client', {})
        local_epochs = int(client_hps.get('local_epochs', 1))
        mu = hps.get('mu', 0.0)

        optimizer = create_optimizer(local_model.parameters(), client_hps)
        scheduler = create_scheduler(optimizer, client_hps, T_max=local_epochs)
        
        global_model_params = [p.detach().clone().to(device) for p in global_model.parameters()]
        
        local_model.train()

        total_train_loss = 0.0
        num_train_batches = 0

        for epoch in range(local_epochs):
            for batch in train_loader:
                optimizer.zero_grad()

                # =================== START OF NLP FIX (CENTRALIZED) ===================
                if isinstance(batch, dict):
                    # --- BRANCH 1: HANDLE TEXT DATA (BERT) ---
                    model_inputs = {k: v.to(device) for k, v in batch.items()}
                    lbls = model_inputs.pop("labels")
                    
                    # Pass the dictionary directly to the full model
                    out = local_model(**model_inputs)
                else:
                    # --- BRANCH 2: HANDLE IMAGE DATA (Your Original Logic) ---
                    imgs, lbls = batch
                    if imgs.shape[0] <= 1:
                        print(f"  --> WARNING: Skipping batch of size {imgs.shape[0]} for Client {cid} to prevent BatchNorm error.")
                        continue
                    imgs, lbls = imgs.to(device), lbls.to(device)
                    out = local_model(imgs)
                # =================== END OF NLP FIX (CENTRALIZED) =====================

                #loss = loss_fn(out, lbls)

                if model_name.lower() in ['charlstm', 'bert']:
                    loss = loss_fn(out.view(-1, out.size(-1)), lbls.view(-1))
                else:
                    loss = loss_fn(out, lbls)


                if mu > 0:
                    prox_term = 0.0
                    for local_param, global_param in zip(local_model.parameters(), global_model_params):
                        prox_term += torch.pow(torch.norm(local_param - global_param), 2)
                    loss += (mu / 2) * prox_term

                total_train_loss += loss.item()
                num_train_batches += 1

                loss.backward()
                clip_grad_norm_(local_model.parameters(), 1.0)
                optimizer.step()
                if scheduler: scheduler.step()

            #if scheduler: scheduler.step()
        
        # Make sure `evaluate_model` is also compatible
        train_acc, _ = evaluate_model(local_model, train_loader, device, loss_fn, model_name)
        test_acc, test_loss = evaluate_model(local_model, val_loader, device, loss_fn, model_name)

        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0

        results['train_loss'].append(avg_train_loss)
        results['test_loss'].append(test_loss)
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)

        # results['train_acc'].append(train_acc)
        # results['test_acc'].append(test_acc)

        print(f"  Client {cid}, Local Epochs {local_epochs}: Train Acc {train_acc:.2f}%, Test Acc {test_acc:.2f}%")

        return local_model.state_dict(), None, len(train_loader.dataset), results
    else:
        raise ValueError(f"Unsupported FL mode: {fl_mode}")



def train_single_client_img_only(model_name, num_classes, arc_cfg, global_model,
                        device, in_channels, train_loader, val_loader, loss_fn,
                        cid, hps: dict, global_epoch: int, fl_mode: str): # Added fl_mode
    """
    Trains a single client model. Adapts based on 'splitfed' or 'centralized' mode.
    """
    results = {'train_loss': [], 'test_acc': [], 'train_acc': []}

    if fl_mode == "splitfed":
        dropout_rate = hps.get('client', {}).get('dropout_rate', 0.1) # Assuming this is for splitfed client
        client_net, server_net, full_ref, _ = create_split_model(
            model_name, num_classes, arc_cfg,
            base_model=global_model, device=device, in_channels=in_channels
        )

        client_hps = hps.get('client', {})
        server_hps = hps.get('server', {})
        mu = hps.get('mu', 0.01) # FedProx mu for split-FL client
        local_epochs = int(client_hps.get('local_epochs', 1))

        opt_c = create_optimizer(client_net.parameters(), client_hps)
        opt_s = create_optimizer(server_net.parameters(), server_hps)

        sch_c = create_scheduler(opt_c, client_hps, T_max=local_epochs)
        sch_s = create_scheduler(opt_s, server_hps, T_max=local_epochs)

        global_client_params = [param.detach().clone() for param in client_net.parameters()]

        client_net.train(); server_net.train()

        for epoch in range(local_epochs):
            for imgs, lbls in train_loader:
                if imgs.shape[0] <= 1:
                    print(f"  --> WARNING: Skipping batch of size {imgs.shape[0]} for Client {cid} to prevent BatchNorm error.")
                    continue
                imgs, lbls = imgs.to(device), lbls.to(device)
                opt_c.zero_grad(); opt_s.zero_grad()
                client_feat = client_net(imgs)
                smashed = client_feat.detach().requires_grad_(True)
                out = server_net(smashed)

                loss = loss_fn(out, lbls)
                if mu > 0:
                    prox_term = 0.0
                    for local_param, global_param in zip(client_net.parameters(), global_client_params):
                        prox_term += torch.pow(torch.norm(local_param - global_param), 2)
                    loss += (mu / 2) * prox_term

                loss.backward()
                client_feat.backward(smashed.grad)
                clip_grad_norm_(client_net.parameters(), 1.0)
                clip_grad_norm_(server_net.parameters(), 1.0)
                opt_c.step(); opt_s.step()

            if sch_c: sch_c.step()
            if sch_s: sch_s.step()

        temp_model = combine_client_server_models(client_net, server_net, full_ref.to(device), device, num_classes, arc_cfg)
        train_acc, _ = evaluate_model(temp_model, train_loader, device, loss_fn)
        test_acc, _ = evaluate_model(temp_model, val_loader, device, loss_fn)

        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)

        print(f"  Client {cid}, Local Epochs {local_epochs}: Train Acc {train_acc:.2f}%, Test Acc {test_acc:.2f}%")

        return client_net.state_dict(), server_net.state_dict(), len(train_loader.dataset), results

    elif fl_mode == "centralized":
        # Create a deep copy of the global model for local client training
        local_model = copy.deepcopy(global_model).to(device)

        # HPs for centralized mode (using client HPs from the agent suggestions)
        client_hps = hps.get('client', {})
        local_epochs = int(client_hps.get('local_epochs', 1))
        mu = hps.get('mu', 0.0) # FedProx mu for centralized FL, typically 0 but can be used

        optimizer = create_optimizer(local_model.parameters(), client_hps)
        scheduler = create_scheduler(optimizer, client_hps, T_max=local_epochs)

        
        # --- MODIFICATION START ---
        # Save initial global model parameters for proximal term calculation if mu > 0
        # IMPORTANT: Ensure these parameters are also on the same device as local_model
        global_model_params = [p.detach().clone().to(device) for p in global_model.parameters()]
        # --- MODIFICATION END ---

        local_model.train() # Set to training mode

        for epoch in range(local_epochs):
            for imgs, lbls in train_loader:
                if imgs.shape[0] <= 1:
                    print(f"  --> WARNING: Skipping batch of size {imgs.shape[0]} for Client {cid} to prevent BatchNorm error.")
                    continue
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()

                out = local_model(imgs)
                loss = loss_fn(out, lbls)

                if mu > 0:
                    prox_term = 0.0
                    for local_param, global_param in zip(local_model.parameters(), global_model_params):
                        prox_term += torch.pow(torch.norm(local_param - global_param), 2)
                    loss += (mu / 2) * prox_term

                loss.backward()
                clip_grad_norm_(local_model.parameters(), 1.0)
                optimizer.step()

            if scheduler: scheduler.step()

        # Evaluate the locally trained full model
        train_acc, _ = evaluate_model(local_model, train_loader, device, loss_fn)
        test_acc, _ = evaluate_model(local_model, val_loader, device, loss_fn)

        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)

        print(f"  Client {cid}, Local Epochs {local_epochs}: Train Acc {train_acc:.2f}%, Test Acc {test_acc:.2f}%")

        # Return full model state_dict in place of client_net.state_dict(), None for server_net
        return local_model.state_dict(), None, len(train_loader.dataset), results
    else:
        raise ValueError(f"Unsupported FL mode: {fl_mode}")




def cluster_fedavg(client_weights, server_weights, client_sizes):
    avg_client = FedAvg(client_weights, client_sizes)
    avg_server = FedAvg(server_weights, client_sizes)
    return avg_client, avg_server


def build_cluster_model(model_name, num_classes, arc_cfg,
                        global_model, device, in_channels,
                        client_weight, server_weight):
    fresh_c, fresh_s, full_ref, _ = create_split_model(
        model_name, num_classes, arc_cfg,
        base_model=global_model, device=device, in_channels=in_channels
    )
    fresh_c.load_state_dict(client_weight)
    fresh_s.load_state_dict(server_weight)

    cluster_model = combine_client_server_models(
        fresh_c, fresh_s, full_ref.to(device),
        device, num_classes, arc_cfg
    )
    return cluster_model

def evaluate_model(model, dataloader, device, loss_fn, model_name=""): # Added model_name for checking
    """
    Evaluates the model's performance on a given dataset.
    Handles both image (tuple) and text (dict) data formats.
    """
    model = model.to(device).eval()
    acc_sum, loss_sum, n = 0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            
            # =================== START OF THE FIX ===================
            
            if isinstance(batch, dict):
                # --- BRANCH 1: HANDLE TEXT DATA (BERT/LSTM) ---
                inputs = {k: v.to(device) for k, v in batch.items()}
                lbls = inputs.pop("labels")
                
                # The full model expects keyword arguments
                out = model(**inputs)
                batch_size = lbls.size(0)

                # --- Reshape tensors for loss and accuracy calculation ---
                # This is the key fix for the shape mismatch error
                loss = loss_fn(out.view(-1, out.size(-1)), lbls.view(-1))
                acc = calculate_accuracy(out.view(-1, out.size(-1)), lbls.view(-1))

            else:
                # --- BRANCH 2: HANDLE IMAGE DATA (Your Original Logic) ---
                imgs, lbls = batch
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                batch_size = imgs.size(0)

                # For images, shapes are already correct
                loss = loss_fn(out, lbls)
                acc = calculate_accuracy(out, lbls)
            
            # =================== END OF THE FIX =====================

            loss_sum += loss.item() * batch_size
            acc_sum += acc * batch_size
            n += batch_size

    if n == 0:
        print(f"  WARNING: No samples processed during evaluation. Returning 0.0 for accuracy and loss.")
        return 0.0, 0.0
    
    return acc_sum / n, loss_sum / n

def evaluate_model_img_only(model, dataloader, device, loss_fn):
    model = model.to(device).eval()
    acc_sum, loss_sum, n = 0, 0, 0
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = model(imgs)
            loss_sum += loss_fn(out, lbls).item() * imgs.size(0)
            acc_sum += calculate_accuracy(out, lbls) * imgs.size(0)
            n += imgs.size(0)

    # --- ADD THIS CHECK TO PREVENT ZeroDivisionError ---
    if n == 0:
        print(f"  WARNING: No samples processed during evaluation. Returning 0.0 for accuracy and loss.")
        return 0.0, 0.0 # Return 0 accuracy and 0 loss if no samples
    # --- END ADDITION ---
    return acc_sum / n, loss_sum / n


def global_fedavg(cluster_models, cluster_sizes):
    return FedAvg(cluster_models, cluster_sizes)

def _format_report(client_states: list) -> list:
    """
    Helper function to format the final YAML report for readability.
    
    MODIFIED to include best performance metrics and history.
    """
    report_data = []
    for i, state in enumerate(client_states):
        # Start with the client ID
        client_report = {'client_id': i}

        # --- NEW: Add the best performance data first for visibility ---
        if state.get('best_accuracy'):
            client_report['best_accuracy'] = state['best_accuracy']
        
        if state.get('best_hps'):
            client_report['best_hps'] = state['best_hps']

        # --- Keep your original fields ---
        client_report['final_search_space'] = state.get('search_space', {})
        
        # --- NEW: Add the full run history for detailed analysis ---
        if state.get('history'):
            client_report['full_run_history'] = state['history']
        
        # Keep the original hpo_report as well, renaming for clarity
        original_hpo_report = {}
        for epoch, report_entry in state.get('hpo_report', {}).items():
            report_key = f"Global Epoch: {epoch}"
            original_hpo_report[report_key] = report_entry
        
        if original_hpo_report:
            client_report['original_hpo_report'] = original_hpo_report

        report_data.append(client_report)
        
    return report_data




def log_global_metrics(
    model_name: str,
    dataset_name: str,
    epoch: int,
    metrics: Dict,

    num_clients: int,
    imbalance_ratio: float,
    global_epochs: int,
    fl_mode: str
):
    """Logs global model performance to a separate CSV file."""
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a new, specific filename for global metrics
    #log_file = os.path.join(results_dir, f"{model_name}__{dataset_name}_global_metrics.csv")
    log_file = os.path.join(
        results_dir,
        f"{model_name}__{dataset_name}_clients{num_clients}_"
        f"imb{imbalance_ratio}_epochs{global_epochs}_{fl_mode}_global_metrics.csv"
    )

    file_exists = os.path.isfile(log_file)
    fieldnames = ['epoch'] + list(metrics.keys())
    
    # 'a' for append mode
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        
        row_to_write = {'epoch': epoch, **metrics}
        writer.writerow(row_to_write)