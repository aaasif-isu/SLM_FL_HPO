import torch
import copy
from ssfl.model_splitter import ResNetBase, CNNBase, BERTBase
from torchvision.models.resnet import ResNet  # Add this import
from collections import OrderedDict
import torch.nn as nn
import re


def FedAvg(weight_dicts, sizes=None):
    """Equal or weighted average of state-dicts."""
    if not weight_dicts:
        raise ValueError("empty FedAvg list")
    if sizes is None:
        sizes = [1] * len(weight_dicts)
    total = sum(sizes)
    w_avg = copy.deepcopy(weight_dicts[0])
    for k in w_avg.keys():
        if w_avg[k].dtype.is_floating_point:
            w_avg[k] *= sizes[0]
            for i in range(1, len(weight_dicts)):
                w_avg[k] += weight_dicts[i][k] * sizes[i]
            w_avg[k] /= total
        # int / bool tensors: keep first client’s copy
    return w_avg

def combine_client_server_models(client_submodel: nn.Module,
                                 server_submodel: nn.Module,
                                 full_template: nn.Module,
                                 device: str,
                                 num_classes: int,
                                 arc_config: int):
    """
    Stitches a client half and server half back into a *single* full network.
    Includes special handling for BERT models to correctly rename state_dict keys.
    """
    client_sd = client_submodel.state_dict()
    server_sd = server_submodel.state_dict()
    merged_sd = OrderedDict()

    # =================== START OF THE FINAL FIX ===================

    if isinstance(full_template, BERTBase):
        # print("Combining BERT models with special key re-indexing...")
        
        # 1. Handle Client Keys (This logic was correct)
        for key, value in client_sd.items():
            new_key = f"bert.{key.replace('encoder_layers', 'encoder.layer')}"
            merged_sd[new_key] = value

        # 2. Handle Server Keys (This is the corrected logic)
        for key, value in server_sd.items():
            if key.startswith('classifier') or key.startswith('fc'):
                merged_sd[key] = value
            elif key.startswith('encoder_layers'):
                # This part correctly calculates the absolute index
                # It finds the number (e.g., '0') in 'encoder_layers.0. ...'
                relative_index = int(re.search(r'\d+', key).group())
                absolute_index = relative_index + arc_config
                
                # Replaces 'encoder_layers.0' with 'encoder.layer.6' (if arc_config=6)
                new_key = key.replace(f'encoder_layers.{relative_index}', f'encoder.layer.{absolute_index}')
                merged_sd[f'bert.{new_key}'] = value
            else:
                # This handles the 'pooler'
                merged_sd[f'bert.{key}'] = value

    else:
        # --- ORIGINAL LOGIC FOR IMAGE MODELS ---
        print("Combining standard (non-BERT) models...")
        merged_sd.update(client_sd)
        merged_sd.update(server_sd)

    # =================== END OF THE FINAL FIX =====================

    full_model = copy.deepcopy(full_template).to(device)
    # Use strict=True (the default) to ensure all keys match perfectly
    full_model.load_state_dict(merged_sd) 

    return full_model


def combine_client_server_models_img_only(client_submodel: nn.Module,
                                 server_submodel: nn.Module,
                                 full_template:  nn.Module,
                                 device: str,
                                 num_classes: int,
                                 arc_config: int):
    """
    Stitch a client half and server half back into a *single* full network.

    Parameters
    ----------
    client_submodel : the trained client slice
    server_submodel : the trained server slice
    full_template   : an un-trained copy of the complete model
                      (CNNBase / ResNetBase / VGGBase) that provides the key
                      layout for .state_dict()
    device          : "cpu" or "cuda:0"
    num_classes     : final classifier size (not used, kept for signature compat)
    arc_config      : split point (needed only for sanity prints)

    Returns
    -------
    nn.Module   —  a full model with merged weights, living on `device`
    """
    # 1. start from the template’s state-dict
    merged_sd = full_template.state_dict()

    # 2. update with client weights (keys already match: layers 0 … arc_config-1)
    merged_sd.update(client_submodel.state_dict())

    # 3. update with server weights
    server_sd = server_submodel.state_dict()

    #    – if your server kept the name "classifier", rename → "fc" here
    # for k in list(server_sd.keys()):
    #     if k.startswith("classifier"):
    #         server_sd[k.replace("classifier", "fc", 1)] = server_sd.pop(k)

    merged_sd.update(server_sd)

    # 4. load back into a *fresh* copy of the template
    full_model = copy.deepcopy(full_template).to(device)
    full_model.load_state_dict(merged_sd)

    return full_model