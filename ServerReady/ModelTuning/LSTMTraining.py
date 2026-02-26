import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm
from LSTMModel import Config, LSTMForecast


def main(local=False):
    """
    Train the LSTM model on the dataset and generate evaluation plots.

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
    # Load Dataset
    # -----------------------------
    dataset = torch.load(dataset_path, weights_only=True)
    encoder_data = dataset['encoder']
    decoder_data = dataset['decoder']
    target_data  = dataset['target']
    full_dataset = TensorDataset(encoder_data, decoder_data, target_data)

    # -----------------------------
    # Train/Val/Test Split (chronological — no data leakage)
    # -----------------------------
    val_ratio, test_ratio = 0.1, 0.1
    n_total = len(full_dataset)
    test_size = int(n_total * test_ratio)
    val_size = int(n_total * val_ratio)
    train_size = n_total - val_size - test_size

    train_dataset = Subset(full_dataset, range(0, train_size))
    val_dataset = Subset(full_dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(full_dataset, range(train_size + val_size, n_total))

    config = Config()

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # -----------------------------
    # Model, Loss, Optimizer
    # -----------------------------
    model = LSTMForecast(config).to(config.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Early stopping
    patience = 50
    best_val_loss = np.inf
    epochs_no_improve = 0

    train_losses, val_losses = [], []

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0
        for enc, dec, tgt in tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}"):
            enc, dec, tgt = enc.to(config.device), dec.to(config.device), tgt.to(config.device)
            optimizer.zero_grad()
            output = model(enc, dec)
            loss = criterion(output, tgt)
            loss.backward()
            # Gradient clipping to prevent gradient explosion with 168-step sequences
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * enc.size(0)

        train_loss = epoch_loss / train_size
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for enc, dec, tgt in val_loader:
                enc, dec, tgt = enc.to(config.device), dec.to(config.device), tgt.to(config.device)
                output = model(enc, dec)
                val_loss_epoch += criterion(output, tgt).item() * enc.size(0)
        val_loss = val_loss_epoch / val_size
        val_losses.append(val_loss)

        # Scheduler step
        scheduler.step(val_loss)

        # Early stopping + save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': vars(config),
                'epoch': epoch,
                'val_loss': val_loss,
            }, model_save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # -----------------------------
    # Load best model
    # -----------------------------
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Training complete. Best model loaded (epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}).")

    # -----------------------------
    # PLOTS
    # -----------------------------
    print("Generating plots...")

    # ---- Collect test predictions (used by multiple plots) ----
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

    # ================================================================
    # Plot 1: Train vs Validation Loss
    # ================================================================
    # FIX: Added markers so single/few data points are visible.
    # Added log scale for when initial loss is much larger than converged loss.
    # Annotated the best epoch with a vertical line.
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, label="Train Loss", linewidth=1.5, marker='o', markersize=2)
    ax.plot(epochs_range, val_losses, label="Validation Loss", linewidth=1.5, marker='o', markersize=2)

    # Mark the best epoch
    best_epoch = checkpoint['epoch']
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best epoch ({best_epoch})')

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Train vs Validation Loss", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Use log scale if the loss range spans more than 1 order of magnitude
    all_losses = train_losses + val_losses
    if max(all_losses) / max(min(all_losses), 1e-10) > 10:
        ax.set_yscale('log')
        ax.set_ylabel("MSE Loss (log scale)", fontsize=12)

    plt.tight_layout()
    plt.savefig(train_val_plot_path, dpi=150)
    plt.close()

    # ================================================================
    # Plot 2: Residual Analysis (Validation Set)
    # ================================================================
    # FIX: X-axis is now PREDICTED (not actual) — standard diagnostic for
    # checking homoscedasticity (whether error variance is constant).
    # Added histogram panel to check if residuals are normally distributed.
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

    # Scatter: residuals vs predicted
    ax1.scatter(preds_val, residuals, alpha=0.15, s=3, color='steelblue')
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax1.set_xlabel("Predicted abvaerk", fontsize=12)
    ax1.set_ylabel("Residual (actual − predicted)", fontsize=12)
    ax1.set_title("Residuals vs Predicted (Validation)", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Histogram of residuals
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

    # ================================================================
    # Plot 3: Example Forecast Windows (Test Set)
    # ================================================================
    # FIX: Replaced the flattened test predictions plot. Flattening all
    # overlapping 168-step windows into one line creates misleading artifacts
    # where the same time period appears multiple times.
    # Instead, show 4 representative individual forecast windows.
    n_test_samples = preds_h.shape[0]
    # Pick 4 evenly spaced windows across the test set
    sample_indices = np.linspace(0, n_test_samples - 1, 4, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        hours = np.arange(1, 169)
        ax.plot(hours, targets_h[idx], label='Actual', linewidth=1.5, color='blue')
        ax.plot(hours, preds_h[idx], label='Predicted', linewidth=1.5, color='red', alpha=0.8)

        # Compute MAE for this window
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

    # ================================================================
    # Plot 4: Per-Horizon Metrics (MSE, MAE, MAPE)
    # ================================================================
    # FIX: Added MAPE (industry standard for energy forecasting).
    # Fixed duplicate label ('144h' was used for 168h).
    # Made vertical day markers consistent across all subplots.
    mse_per_horizon = np.mean((preds_h - targets_h) ** 2, axis=0)
    mae_per_horizon = np.mean(np.abs(preds_h - targets_h), axis=0)

    # MAPE per horizon (skip zeros in actuals)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_per_sample = np.abs((targets_h - preds_h) / targets_h) * 100
        mape_per_sample = np.where(np.isfinite(mape_per_sample), mape_per_sample, np.nan)
    mape_per_horizon = np.nanmean(mape_per_sample, axis=0)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    hours = range(1, 169)

    # Day markers — consistent across all subplots
    day_hours = [24, 48, 72, 96, 120, 144, 168]
    day_labels = ['1d', '2d', '3d', '4d', '5d', '6d', '7d']

    def add_day_markers(ax):
        for h, lbl in zip(day_hours, day_labels):
            ax.axvline(x=h, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            ax.annotate(lbl, xy=(h, 1.02), xycoords=('data', 'axes fraction'),
                       ha='center', fontsize=8, color='gray')

    # MSE
    ax1.plot(hours, mse_per_horizon, color='blue', linewidth=1.2)
    ax1.set_ylabel("MSE", fontsize=12)
    ax1.set_title("Per-Horizon Forecast Error (Test Set)", fontsize=14)
    ax1.grid(True, alpha=0.3)
    add_day_markers(ax1)

    # MAE
    ax2.plot(hours, mae_per_horizon, color='red', linewidth=1.2)
    ax2.set_ylabel("MAE", fontsize=12)
    ax2.grid(True, alpha=0.3)
    add_day_markers(ax2)

    # MAPE
    ax3.plot(hours, mape_per_horizon, color='green', linewidth=1.2)
    ax3.set_xlabel("Forecast Horizon (hours)", fontsize=12)
    ax3.set_ylabel("MAPE (%)", fontsize=12)
    ax3.grid(True, alpha=0.3)
    add_day_markers(ax3)

    plt.tight_layout()
    plt.savefig(horizon_plot_path, dpi=150)
    plt.close()

    print("Plots saved successfully.")


# Allow standalone execution: python3 LSTMTraining.py --local
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', help='Use local relative paths instead of server paths')
    args = parser.parse_args()
    main(local=args.local)
