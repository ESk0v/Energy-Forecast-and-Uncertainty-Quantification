# HyperParameterTuning/ArchitectureTuning.py
import torch
import torch.nn as nn
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import Config

# ----------------------------
# Grid Search Space
# ----------------------------
HIDDEN_SIZES = [32, 64, 128, 256]
NUM_LAYERS_OPTIONS = [1, 2, 3]
TUNE_EPOCHS = 300

def run_architecture_search(model_class, model_kwargs_template, train_loader, val_loader, criterion, device):
    """
    Grid search over hidden_size and num_layers combinations.
    Returns (best_hidden_size, best_num_layers, results_dict)
    """
    combos = list(itertools.product(HIDDEN_SIZES, NUM_LAYERS_OPTIONS))
    total  = len(combos)
    print(f"\nStarting Architecture Grid Search — {total} configurations, {TUNE_EPOCHS} epochs each\n")

    results = {}

    for i, (hidden_size, num_layers) in enumerate(combos, start=1):
        label = f"H{hidden_size}_L{num_layers}"

        # --- Progress indicator ---
        bar_filled = int((i - 1) / total * 20)
        bar_str    = "█" * bar_filled + "░" * (20 - bar_filled)
        print(f"  [{bar_str}] ({i}/{total}) Testing {label}...", end=" ", flush=True)

        model_kwargs = {
            **model_kwargs_template,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
        }

        val_loss = _evaluate_config(
            model_class=model_class,
            model_kwargs=model_kwargs,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        results[label] = {
            "val_loss": val_loss,
            "hidden_size": hidden_size,
            "num_layers": num_layers
        }
        print(f"Val Loss: {val_loss:.6f}")

    # Final completed bar
    print(f"  [{'█' * 20}] ({total}/{total}) Done!\n")

    # --- Find best ---
    best_label  = min(results, key=lambda k: results[k]["val_loss"])
    best_hidden = results[best_label]["hidden_size"]
    best_layers = results[best_label]["num_layers"]

    print(f"Best config: hidden_size={best_hidden}, num_layers={best_layers} "
          f"(val_loss={results[best_label]['val_loss']:.6f})")

    return best_hidden, best_layers, results


def _evaluate_config(model_class, model_kwargs, train_loader, val_loader, criterion, device):
    """
    Train a model config for TUNE_EPOCHS and return the best validation loss seen.
    """
    model = model_class(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(TUNE_EPOCHS):
        # --- Train ---
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_losses.append(criterion(model(X_val), y_val).item())

        avg_val = np.mean(val_losses)
        if avg_val < best_val_loss:
            best_val_loss = avg_val

        # --- Epoch progress (overwrites same line) ---
        epoch_bar_filled = int((epoch + 1) / TUNE_EPOCHS * 20)
        epoch_bar_str    = "█" * epoch_bar_filled + "░" * (20 - epoch_bar_filled)
        print(f"    Epoch [{epoch_bar_str}] ({epoch+1}/{TUNE_EPOCHS})  "
              f"Val Loss: {avg_val:.6f}  Best: {best_val_loss:.6f}",
              end="\r", flush=True)

    # Print a newline after the epoch bar finishes so the outer progress bar prints cleanly
    print()

    return best_val_loss


def plot_architecture_results(results: dict, save_path):
    """
    Bar chart of validation loss per config, sorted ascending.
    Best config is highlighted in gold.
    """
    # Sort by val_loss ascending
    sorted_items = sorted(results.items(), key=lambda x: x[1]["val_loss"])
    labels = [item[0] for item in sorted_items]
    losses = [item[1]["val_loss"] for item in sorted_items]

    colors = ["gold" if i == 0 else "steelblue" for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(labels, losses, color=colors, edgecolor="black", width=0.6)

    # Annotate each bar with its value
    for bar, loss in zip(bars, losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(losses) * 0.005,
            f"{loss:.5f}",
            ha="center", va="bottom", fontsize=8
        )

    gold_patch = mpatches.Patch(color="gold", label="Best config")
    ax.legend(handles=[gold_patch])
    ax.set_xlabel("Configuration  (HiddenSize _ NumLayers)")
    ax.set_ylabel("Best Validation Loss")
    ax.set_title("Grid Search — Architecture & Hidden Size Tuning")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Plot saved to {save_path}")