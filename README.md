# ME5406 Project 2: Reinforcement Learning for Narrow-Passage Path Planning with a Manipulator

## Overview
This repository contains the source code, pre-trained model weights, and evaluation scripts for our ME5406 Reinforcement Learning project. It demonstrates a Deep Reinforcement Learning approach—evaluating algorithms such as PPO, SAC, and TD3—to solve narrow-passage path planning for a UR10e robotic arm. Built on NVIDIA's Isaac Sim, the codebase provides the complete environment formulation and training pipeline to evaluate an agent capable of real-time, online trajectory adjustments to safely navigate through constrained window obstacles.

## Repository Structure & Main Files
* **`TD3/`, `PPO/`, `SAC/`**: These directories contain the core training scripts and algorithm-specific network implementations for each Reinforcement Learning method we evaluated. 
* **`common/`**: Contains shared scripts, utilities, and configurations utilized across all algorithms (e.g., Isaac Sim environment wrappers, unified reward functions).
* **`eval_actor_only.py`**: The primary validation and testing script that demonstrates the functioning of the model in the test environment.
* **`best_checkpoints/`**: Directory containing the fully-trained and functional `.pt` model weights.
* **`old_version_checkpoints/`**: A repository of intermediate model weights saved during the iterative training and tuning process.
* **`requirements.txt`**: Defines the exact dependencies needed to execute the codebase. 

## How to Use This Repository

### 1. Environment Setup
This project requires a specific Conda environment to seamlessly run the Isaac Sim physics simulation alongside the RL training and tracking libraries.

Follow these steps to configure the environment:
```bash
# 1. Create and activate the specific conda environment
conda create -n isaac_env python=3.10
conda activate isaac_env

# 2. Install Isaac Sim with the necessary extensions
pip install "isaacsim[all,extscache]" --extra-index-url [https://pypi.nvidia.com](https://pypi.nvidia.com)

# 3. Install additional required libraries for tracking and visualization
pip install matplotlib tensorboard
```

### 2. Training a New Agent
To train an agent from scratch, navigate to the specific algorithm's directory and execute its training script:
```bash
cd AlgoName
python train_AlgoName.py # Replace with the exact script name inside the folder if different
```

### 3. Evaluation / Testing
To evaluate a fully-trained functional model, you can use the provided validation script to visualize the functioning of the trained RL agent. 

```bash
# Run the evaluation script for the trained someone Algo agent over 20 episodes
python eval_actor_only.py --checkpoint best_checkpoints/best_for_AlgoName.pt --episodes 20
```

## Disclaimers & Acknowledgements
* **External Code Usage:** Any external code or open-source templates used as a starting point for this project have been fully disclosed in the individual project report. At least 50% of the total project's codebase was written from scratch by the group, adhering strictly to the ME5406 assignment rules.
* **Generative AI Usage:** Generative AI tools were utilized strictly within NUS guidelines to assist with boilerplate code structuring and documentation formatting. All use cases are fully documented within the respective individual reports.
