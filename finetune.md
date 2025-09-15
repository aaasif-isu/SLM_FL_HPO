# Fine-tuning Instructions

This document explains the process of preparing datasets and fine-tuning Small Language Models (SLMs) for per-client hyperparameter optimization (HPO) in our framework. The goal is to adapt SLMs so that they can generate hyperparameter configurations in a structured JSON format, based on data collected from LLM-based optimization.

***

### Step 1: Prepare the JSON File for HP Fine-tuning

1.  Navigate to the **parse** folder:
    ```bash
    cd code/parse/
    ```
2.  Run the script to generate the JSON file:
    ```bash
    python create_json_file_v2.py
    ```
    * **Input**: Output of the LLM API calls for hyperparameter optimization.
    * **Output**: `epoch_client_hps.json`

***

### Step 2: Create Supervised Fine-tuning (SFT) Dataset

We need the dataset in a specific JSON format so that the SLM can be fine-tuned properly.

1.  Run the following script:
    ```bash
    python sft_json.py
    ```
    This will create two datasets:
    * `sft_instruct.jsonl` → used for fine-tuning
    * `sft_chat.jsonl` → optional, chat-style format

***

### Step 3: Fine-tune the SLM (Qwen 0.5B Example)

1.  Navigate to the **finetune** folder:
    ```bash
    cd code/finetune/
    ```
2.  Run the fine-tuning script:
    ```bash
    python sft_qwen.py
    ```
    * **Model**: Qwen 0.5B
    * **Dataset**: `sft_instruct.jsonl`
    * **Output**: Training log showing the loss of both the base model and the fine-tuned model.

***

### Step 4: Test the Fine-tuned Model

1.  After training, test the model by running:
    ```bash
    python sft_inference.py
    ```
    * This script compares outputs of the base model and the fine-tuned model on a specific prompt.
    * It ensures the fine-tuned model generates hyperparameters in the required JSON format.

***

### Summary of Workflow

* `create_json_file_v2.py` → Generate raw hyperparameter dataset.
* `sft_json.py` → Convert dataset into SFT format (`sft_instruct.jsonl`).
* `sft_qwen.py` → Fine-tune the Qwen 0.5B model.
* `sft_inference.py` → Compare outputs of base vs. fine-tuned model.

This pipeline ensures that the SLM learns to generate high-quality hyperparameters tailored for federated learning clients.