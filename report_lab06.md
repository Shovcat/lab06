# Lab 6: Reproducible YAML-Driven PyTorch Pipeline & MLflow Experiment Tracking

## 1. Objective & Pipeline Architecture

The goal of this lab is to transition standalone model scripts into a modular, reproducible machine learning pipeline. All experiment configurations, hyperparameters, and system paths are decoupled from the Python execution logic via `config.yaml` and CLI overrides, while metrics, model checkpoints, and hardware benchmarks are systematically logged using **MLflow** (`sqlite:///mlflow.db`).

### Key Engineering Components

- **Configuration Management (`config.yaml` & `argparse`):** Centralizes random seeds (`seed: 42`), batch sizes (`128`), learning rates (`0.001`), and epoch counts, while supporting CLI overrides (`--model MLP/CNN/Attention`) for clean comparative runs.
- **Automated Checkpointing:** Evaluates validation accuracy at the end of every epoch, saves the highest-performing model weights locally to `./checkpoints/best_<model>_model.pth`, and logs the serialized `.pth` file directly as an MLflow artifact.
- **Hardware & Latency Profiling:** Uses `torchinfo` and `thop` to record parameter counts and FLOPs, alongside a synchronized `time.perf_counter()` benchmark measuring median inference latency across single-sample (`B=1`) and batch (`B=128`) workloads.

---

## 2. Reproducibility & Execution Workflow

1. **Install Dependencies:**
   ```bash
   pip install torch torchvision pyyaml mlflow torchinfo thop numpy
   ```
