# utils_seed.py
import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42):
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducible results.
    Also enforces deterministic backend behavior in CUDA.
    """
    # Basic seeds
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[Seed] All random generators initialized with seed = {seed}")
