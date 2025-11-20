# code/eval/eval_fl.py

import os
import sys
import argparse
import yaml
import torch

# Make repo root (code/) importable when running from inside eval/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(THIS_DIR)
sys.path.append(CODE_ROOT)

from ssfl.utils import load_dataset, subsample_dataset
from ssfl.model_splitter import create_global_model
from ssfl.trainer_utils import evaluate_model
from ssfl.utils_seed import seed_everything


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="eval/eval_config.yaml",
        help="Eval config path (relative to code/).",
    )
    args = parser.parse_args()

    # paths in YAML are relative to code/, so we keep cwd=code
    cfg = load_yaml(args.config)
    fl_cfg = cfg["eval"]["fl"]

    if not fl_cfg.get("enabled", True):
        print("[eval_fl] FL evaluation disabled in config.")
        return

    use_lora = bool(fl_cfg.get("use_lora", False))

    if use_lora:
        ckpt_path = fl_cfg["ckpt_lora"]
    else:
        ckpt_path = fl_cfg["ckpt_nolora"]

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # ---- Read eval config directly (no training config) ----
    model_name = fl_cfg["model_name"]
    dataset_name = fl_cfg["dataset_name"]
    test_fraction = float(fl_cfg.get("test_sample_fraction", 1.0))
    eval_batch_size = int(fl_cfg.get("eval_batch_size", 128))
    seed = int(fl_cfg.get("seed", 42))
    device_str = fl_cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    seed_everything(seed)

    train_dataset, test_dataset, num_classes, image_size, in_channels = load_dataset(
        dataset_name
    )
    print(f"[eval_fl] Dataset: {dataset_name.upper()}")

    # For evaluation we only care about test set
    test_subset = subsample_dataset(test_dataset, test_fraction)

    test_loader = torch.utils.data.DataLoader(
        test_subset, batch_size=eval_batch_size, shuffle=False, drop_last=True
    )

    global_model = create_global_model(
        model_name, num_classes, in_channels, device
    )

    print(f"[eval_fl] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    global_model.load_state_dict(state)

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    acc, loss = evaluate_model(
        global_model, test_loader, device, loss_fn, model_name=model_name
    )
    tag = "LoRA-FL" if use_lora else "Baseline-FL"
    print(
        f"[eval_fl] [{tag}] Test Accuracy: {acc:.2f}%  |  Test Loss: {loss:.4f}"
    )


if __name__ == "__main__":
    main()
