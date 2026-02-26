"""
Plotting — Evaluation plots for the trained LSTM model.
=========================================================
Loads the saved checkpoint and dataset, rebuilds the model, runs inference
on the validation and test sets, and generates all evaluation plots.

Can be run independently after training is complete:
    python3 Plotting.py --local
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from LSTMModel import Config, LSTMForecast


def main(local=False):
    """
    Generate all evaluation plots from a trained model checkpoint.

    Args:
        local: If True, use relative paths. If False, use server paths.
    """

    # -----------------------------
    # Paths
    # -----------------------------
    if local:
        _dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(_dir, "dataset.pt")
        model_save_path = os.path.join(_dir, "best_lstm_forecast_model.pth")
        train_val_plot_path = os.path.join(_dir, "train_val_loss.png")
        residual_plot_path = os.path.join(_dir, "residuals.png")
        test_plot_path = os.path.join(_dir, "test_predictions.png")
        horizon_plot_path = os.path.join(_dir, "per_horizon_metrics.png")
        print("Running in LOCAL mode (relative paths)")
    else:
        dataset_path = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/dataset.pt"
        model_save_path = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/best_lstm_forecast_model.pth"
        train_val_plot_path = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/train_val_loss.png"
        residual_plot_path = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/residuals.png"
        test_plot_path = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/test_predictions.png"
        horizon_plot_path = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/per_horizon_metrics.png"
        print("Running in SERVER mode (absolute paths)")

    # -----------------------------
    # Load checkpoint
    # -----------------------------
    checkpoint = torch.load(model_save_path, map_location='cpu')
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    best_epoch = checkpoint['epoch']
    print(f"Checkpoint loaded (best epoch: {best_epoch}, val_loss: {checkpoint['val_loss']:.4f})")

    # -----------------------------
    # Load dataset and rebuild data loaders
    # -----------------------------
    dataset = torch.load(dataset_path, weights_only=True)
    encoder_data = dataset['encoder']
    decoder_data = dataset['decoder']
    target_data = dataset['target']
    full_dataset = TensorDataset(encoder_data, decoder_data, target_data)

    val_ratio, test_ratio = 0.1, 0.1
    n_total = len(full_dataset)
    test_size = int(n_total * test_ratio)
    val_size = int(n_total * val_ratio)
    train_size = n_total - val_size - test_size

    val_dataset = Subset(full_dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(full_dataset, range(train_size + val_size, n_total))

    config = Config()
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # -----------------------------
    # Rebuild and load model
    # -----------------------------
    model = LSTMForecast(config).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # -----------------------------
    # Collect test predictions (used by multiple plots)
    # -----------------------------
    all_preds_h, all_targets_h = [], []
    with torch.no_grad():
        for enc, dec, tgt in test_loader:
            enc, dec = enc.to(config.device), dec.to(config.device)
            output = model(enc, dec)
            all_preds_h.append(output.cpu().numpy())
            all_targets_h.append(tgt.numpy())

    # Shape: (n_samples, 168) — each row is one 168-hour forecast window
    preds_h = np.concatenate(all_preds_h, axis=0)
    targets_h = np.concatenate(all_targets_h, axis=0)

    print(f"Generating plots ({preds_h.shape[0]} test samples)...")

    # ================================================================
    # Plot 1: Train vs Validation Loss
    # ================================================================
    # Markers ensure visibility even with few epochs.
    # Auto log-scale when loss range spans > 1 order of magnitude.
    # Best epoch annotated with a vertical line.
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, label="Train Loss", linewidth=1.5, marker='o', markersize=2)
    ax.plot(epochs_range, val_losses, label="Validation Loss", linewidth=1.5, marker='o', markersize=2)

    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best epoch ({best_epoch})')

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Train vs Validation Loss", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    all_losses = train_losses + val_losses
    if len(all_losses) > 0 and max(all_losses) / max(min(all_losses), 1e-10) > 10:
        ax.set_yscale('log')
        ax.set_ylabel("MSE Loss (log scale)", fontsize=12)

    plt.tight_layout()
    plt.savefig(train_val_plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {train_val_plot_path}")

    # ================================================================
    # Plot 2: Residual Analysis (Validation Set)
    # ================================================================
    # X-axis is PREDICTED (not actual) — standard diagnostic for checking
    # homoscedasticity (whether error variance is constant).
    # Second panel: histogram to check if residuals are normally distributed.
    all_preds_val, all_targets_val = [], []
    with torch.no_grad():
        for enc, dec, tgt in val_loader:
            enc, dec = enc.to(config.device), dec.to(config.device)
            output = model(enc, dec)
            all_preds_val.append(output.cpu())
            all_targets_val.append(tgt.cpu())

    preds_val = torch.cat(all_preds_val).numpy().flatten()
    targets_val = torch.cat(all_targets_val).numpy().flatten()
    residuals = targets_val - preds_val

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(preds_val, residuals, alpha=0.15, s=3, color='steelblue')
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax1.set_xlabel("Predicted abvaerk", fontsize=12)
    ax1.set_ylabel("Residual (actual − predicted)", fontsize=12)
    ax1.set_title("Residuals vs Predicted (Validation)", fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.hist(residuals, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
    ax2.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_xlabel("Residual", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Residual Distribution", fontsize=14)
    mean_r, std_r = np.mean(residuals), np.std(residuals)
    ax2.legend([f"Mean: {mean_r:.4f}\nStd: {std_r:.4f}"], fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(residual_plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {residual_plot_path}")

    # ================================================================
    # Plot 3: Example Forecast Windows (Test Set)
    # ================================================================
    # Shows 4 representative individual 168-hour forecasts evenly spaced
    # across the test set, instead of a misleading flattened line.
    n_test_samples = preds_h.shape[0]
    sample_indices = np.linspace(0, n_test_samples - 1, 4, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        hours = np.arange(1, 169)
        ax.plot(hours, targets_h[idx], label='Actual', linewidth=1.5, color='blue')
        ax.plot(hours, preds_h[idx], label='Predicted', linewidth=1.5, color='red', alpha=0.8)

        window_mae = np.mean(np.abs(targets_h[idx] - preds_h[idx]))
        ax.set_title(f"Test Sample {idx} (MAE: {window_mae:.4f})", fontsize=12)
        ax.set_xlabel("Forecast Hour", fontsize=10)
        ax.set_ylabel("abvaerk", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Example 168-Hour Forecast Windows (Test Set)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(test_plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {test_plot_path}")

    # ================================================================
    # Plot 4: Per-Horizon Metrics (MSE, MAE, MAPE)
    # ================================================================
    # Shows how forecast accuracy degrades with horizon distance.
    # Day markers (1d–7d) are consistent across all subplots.
    mse_per_horizon = np.mean((preds_h - targets_h) ** 2, axis=0)
    mae_per_horizon = np.mean(np.abs(preds_h - targets_h), axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        mape_per_sample = np.abs((targets_h - preds_h) / targets_h) * 100
        mape_per_sample = np.where(np.isfinite(mape_per_sample), mape_per_sample, np.nan)
    mape_per_horizon = np.nanmean(mape_per_sample, axis=0)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    hours = range(1, 169)

    day_hours = [24, 48, 72, 96, 120, 144, 168]
    day_labels = ['1d', '2d', '3d', '4d', '5d', '6d', '7d']

    def add_day_markers(ax):
        for h, lbl in zip(day_hours, day_labels):
            ax.axvline(x=h, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            ax.annotate(lbl, xy=(h, 1.02), xycoords=('data', 'axes fraction'),
                        ha='center', fontsize=8, color='gray')

    ax1.plot(hours, mse_per_horizon, color='blue', linewidth=1.2)
    ax1.set_ylabel("MSE", fontsize=12)
    ax1.set_title("Per-Horizon Forecast Error (Test Set)", fontsize=14)
    ax1.grid(True, alpha=0.3)
    add_day_markers(ax1)

    ax2.plot(hours, mae_per_horizon, color='red', linewidth=1.2)
    ax2.set_ylabel("MAE", fontsize=12)
    ax2.grid(True, alpha=0.3)
    add_day_markers(ax2)

    ax3.plot(hours, mape_per_horizon, color='green', linewidth=1.2)
    ax3.set_xlabel("Forecast Horizon (hours)", fontsize=12)
    ax3.set_ylabel("MAPE (%)", fontsize=12)
    ax3.grid(True, alpha=0.3)
    add_day_markers(ax3)

    plt.tight_layout()
    plt.savefig(horizon_plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {horizon_plot_path}")

    print("All plots saved successfully.")


# Allow standalone execution: python3 Plotting.py --local
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', help='Use local relative paths instead of server paths')
    args = parser.parse_args()
    main(local=args.local)

