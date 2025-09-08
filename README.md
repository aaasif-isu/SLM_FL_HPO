### New README.md

# Agent-Based Per-Client Hyperparameter Optimization in Federated Learning

This project introduces a novel framework for performing dynamic, per-client hyperparameter optimization (HPO) within a Federated Learning (FL) environment. It addresses the limitations of traditional FL, where a single, fixed set of hyperparameters is applied to all clients, often leading to suboptimal performance in heterogeneous settings where client data and resources vary significantly.

The core of this framework is an intelligent, agent-based system that leverages the reasoning capabilities of Large Language Models (LLMs) to individually tune hyperparameters for each client. This approach allows the model to adapt to each client's unique data distribution and performance over time, moving beyond rigid, static configurations.

## Core Components and Methodology

The framework is designed for a "tuning-while-training" paradigm, which enables efficient HPO without the need for expensive, separate tuning phases. This process is powered by a dual-agent system and several key architectural decisions:

* **LLM-Powered Agents:** The system uses two main agents powered by an LLM:
    * **The HPO Agent (Suggester):** This agent analyzes a client's full context, including its cluster ID, performance history, and a personalized search space. It then prompts an LLM to suggest an optimal set of hyperparameters for the upcoming local training round.
    * **The Analyzer Agent:** After a client completes its training, this agent evaluates the results (e.g., training vs. test accuracy, signs of overfitting). It then uses an LLM to provide a structured analysis and propose a *new, refined search space* for that client. This creates an adaptive, persistent learning loop that improves a client's configuration over successive rounds.
* **Split Federated Learning (SplitFed):** The architecture supports SplitFed, where the neural network is divided between the client (front layers) and the server (back layers). This reduces the computational load on client devices.
* **`FedProx` for Stability:** To counteract model divergence caused by clients training with different hyperparameters, the framework employs the `FedProx` algorithm. By adding a proximal term to the client's loss function, it ensures the stability of the global model's convergence.


## Key Features

* **Per-Client HPO:** Dynamically tunes hyperparameters for each client, recognizing their unique characteristics.
* **Adaptive Search Space:** The hyperparameter search space for each client evolves over time based on its specific performance and the analyzer agent's recommendations.
* **Heterogeneous Environment Support:** The `resource_profiler.py` script simulates client latency and assigns them to different resource-based clusters, enabling experiments in heterogeneous FL settings.
* **Robust Training:** The framework implements `FedProx` to ensure stable global model convergence in heterogeneous environments.
* **Comprehensive Reporting:** The system generates a detailed `client_hpo_states.yaml` file, which provides full transparency into the HPO process by logging each client's suggested hyperparameters, performance, and the final refined search space.

## Project Structure

The project's code is organized to separate the core FL logic from the agent-based HPO components.

* `code/ssfl/`: This directory contains the foundational FL implementation.
    * `model_splitter.py`: Defines the logic for splitting models into client and server components.
    * `aggregation.py`: Implements `FedAvg` and functions for combining split models.
    * `trainer.py` and `trainer_utils.py`: Orchestrate the main training loop, client selection, and evaluation.
    * `strategies.py`: Implements  HPO strategies, 
* `code/agent/`: This directory contains the LLM-powered HPO agents.
    * `hp_agent.py`: The "Suggester" agent that recommends new hyperparameters.
    * `analyzer_agent.py`: The agent that refines the search space based on client performance.
    * `workflow.py`: Defines the `langgraph` workflow that orchestrates the agents and a background CPU worker thread.
    * `llm_api.py`: Handles API calls to the LLM.
* `code/config/`: Contains various YAML configuration files for different experiments.
* `code/main.py`: The primary entry point for running experiments.
* `code/run.sh`: A template Slurm script for running experiments on a computing cluster.

### Instructions to Replicate the Project

To replicate the project, follow these steps to set up the environment, configure your experiments, and run the code.

#### 1. Setup and Installation

1.  **Clone the Repository:** Clone the project from its source.
2.  **Create a Virtual Environment:** It is highly recommended to use a virtual environment. The `run.sh` script suggests using `conda`.
    ```bash
    conda create -n agent-hpo python=3.10
    conda activate agent-hpo
    ```
3.  **Install Dependencies:** Navigate to the `code/` directory and install the required Python packages using the provided `requirements.txt` file.
    ```bash
    cd .../agentflhpo/AgentFLHPO-main/code
    pip install -r requirements.txt
    ```
4.  **Set Up the LLM API Key:** The project uses the `openrouter.ai` API to access an LLM. You must obtain an API key from OpenRouter and set it as an environment variable.
    * Create a `.env` file in the `code/agent/` directory.
    * Add your API key to the file:
        ```
        OPENROUTER_API_KEY="sk-..."
        ```
    * The `llm_api.py` file will automatically load this key.

#### 2. Configuration

All experimental settings are controlled via YAML configuration files located in `code/config/` and the main `code/model_config.yaml` file.

* **`model_config.yaml`**: This file is the primary configuration file and should be modified to define your experiment. Key parameters include:
    * `model_name`: The neural network architecture (e.g., `ResNet18`, `charlstm`).
    * `dataset_name`: The dataset to use (e.g., `cifar10`, `shakespeare`, `pacs`). The first run will automatically download the specified dataset to a `data/` directory.
    * `num_clients`: The total number of clients.
    * `global_epochs`: The number of global communication rounds.
    * `fl_mode`: The Federated Learning mode, either `splitfed` or `centralized`.
    * `hpo_strategy.method`: The HPO method to use (`agent` only).

#### 3. Running an Experiment

There are two primary ways to run an experiment: locally or on a Slurm cluster.

**Local Execution**

1.  Navigate to the `code/` directory.
2.  Run the `main.py` script, specifying the configuration file you wish to use. The following command will run with the default `model_config.yaml` file.

    ```bash
    python main.py 
    ```
    *Note: The experiment's output is redirected to a log file. You can view the log file in the `logs/` directory.*

**Slurm Cluster Execution**

The `run.sh` script is a template designed to be submitted to a Slurm-managed cluster.

1.  **Configure the Script:**
    * Edit the `run.sh` file to set your account name, virtual environment name, and the absolute path to your project directory.
    * Update the `CONFIGS` array with the paths to the specific experiment YAML files you want to run.
2.  **Submit the Job:** Submit the script using the `sbatch` command.
    ```bash
    sbatch run.sh
    ```
    *Note: This script is configured as a job array, allowing you to run multiple experiments in parallel.*

#### 4. Understanding the Output

Upon completion, the project generates several key files to help you analyze the results:

* `client_hpo_states.yaml`: A detailed report of the HPO process for every client, including their final search spaces and historical performance.
* `[model_name]_[dataset_name]_..._training_details_metrics.csv`: A CSV file containing epoch-by-epoch training and performance metrics for each client, including training time, LLM latency, and token usage.
* `[model_name]_[dataset_name]_..._global_metrics.csv`: A CSV file that logs the global model's performance (accuracy and loss) after each federated round.

These files are saved in the `results/` and `logs/` directories, providing a complete record of each experiment.