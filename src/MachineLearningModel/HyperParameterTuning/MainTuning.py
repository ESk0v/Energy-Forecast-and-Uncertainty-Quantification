# HyperParameterTuning/MainTuning.py
import sys
import os

# Add parent folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import re

# Now relative imports can become absolute
import Config
from LearningRateTuning import lr_range_test, plot_lr_results
from ArchitectureTuning import run_architecture_search, plot_architecture_results
from Data import load_dataset
from LSTMModel import LSTMModel


# --- Device ---
if Config.DEVICE.lower() == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Config.DEVICE is set to 'cuda' but CUDA is not available. "
            "Install CUDA-enabled PyTorch or change Config.DEVICE to 'cpu'."
        )
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def update_config_file(new_lr):
    update_config_value(
        r"LEARNING_RATE\s*=\s*[\d\.eE+-]+",
        f"LEARNING_RATE = {new_lr:.6f}"
    )
    print(f"Config file updated: LEARNING_RATE = {new_lr:.6f}")


def update_config_architecture(hidden_size: int, num_layers: int):
    update_config_value(
        r"HIDDEN_SIZE\s*=\s*\d+",
        f"HIDDEN_SIZE = {hidden_size}"
    )
    update_config_value(
        r"NUM_LAYERS\s*=\s*\d+",
        f"NUM_LAYERS = {num_layers}"
    )
    print(f"Config file updated: HIDDEN_SIZE = {hidden_size}, NUM_LAYERS = {num_layers}")


# ============================================================
# Shared data loading helper
# ============================================================
def _build_loaders():
    """
    Load dataset and return (train_loader, val_loader).
    val_loader is carved from the test split to mirror Config settings.
    """
    Samples, Targets, _ = load_dataset(Config.DATASET_PATH, False)

    X = torch.tensor(Samples, dtype=torch.float32)
    y = torch.tensor(Targets, dtype=torch.float32).view(-1, 1)

    dataset = torch.utils.data.TensorDataset(X, y)
    total = len(dataset)

    test_size  = int(total * Config.TEST_SIZE)
    train_size = total - test_size
    val_size   = int(test_size * Config.VAL_SPLIT)
    test_size  = test_size - val_size

    train_set, val_set, _ = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=Config.BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=Config.BATCH_SIZE, shuffle=False
    )

    return train_loader, val_loader


# ============================================================
# LR Tuning
# ============================================================
def run_lr_tuning():
    """Run learning rate tuning using LRRT."""
    train_loader, _ = _build_loaders()

    model_kwargs = {
        "input_size": Config.INPUT_SIZE,
        "hidden_size": Config.HIDDEN_SIZE,
        "num_layers": Config.NUM_LAYERS,
        "dropout": Config.DROPOUT,
    }

    criterion = nn.MSELoss()

    lrs, losses, best_lr = lr_range_test(
        model_class=LSTMModel,
        model_kwargs=model_kwargs,
        train_loader=train_loader,
        criterion=criterion,
        device=device
    )

    plot_lr_results(lrs, losses, save_path=Config.LRRT_PLOT_PATH, title="Learning Rate Range Test")

    Config.LEARNING_RATE = best_lr
    print(f"\nConfig updated in memory: LEARNING_RATE = {Config.LEARNING_RATE:.6f}")
    update_config_file(best_lr)


# ============================================================
# Architecture Tuning
# ============================================================
def run_architecture_tuning():
    """Run grid search over hidden_size and num_layers."""
    train_loader, val_loader = _build_loaders()

    # Template — hidden_size and num_layers are overridden inside the search
    model_kwargs_template = {
        "input_size": Config.INPUT_SIZE,
        "dropout": Config.DROPOUT,
    }

    criterion = nn.MSELoss()

    best_hidden, best_layers, results = run_architecture_search(
        model_class=LSTMModel,
        model_kwargs_template=model_kwargs_template,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device
    )

    plot_architecture_results(results, save_path=Config.GRID_SEARCH_PLOT_PATH)

    # Update memory
    Config.HIDDEN_SIZE = best_hidden
    Config.NUM_LAYERS  = best_layers
    print(f"\nConfig updated in memory: HIDDEN_SIZE = {best_hidden}, NUM_LAYERS = {best_layers}")

    # Update file
    update_config_architecture(best_hidden, best_layers)


# ============================================================
# Entry point — runs all tuning in logical order
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Step 1: Learning Rate Tuning (LRRT)")
    print("=" * 50)
    run_lr_tuning()

    print("\n" + "=" * 50)
    print("Step 2: Architecture Tuning (Grid Search)")
    print("=" * 50)
    run_architecture_tuning()