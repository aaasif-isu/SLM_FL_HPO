# /u/aalasif/SLM_FL_HPO/code/finetune/eval_loss.py
import torch
from torch.utils.data import DataLoader
from typing import Callable, Optional

def evaluate_loss(model, dataset, batch_size=2, collate_fn: Optional[Callable]=None):
    """
    Compute average loss of a model on a tokenized dataset.

    - dataset items must contain 'input_ids', 'attention_mask', 'labels'
    - set collate_fn to the SAME collator used in training (pads to tensors)
    """
    model.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    losses = []
    for batch in dl:
        # Your collator returns torch tensors, so we can move directly to device:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
        losses.append(loss.item())
    return sum(losses)/len(losses) if losses else float("nan")
