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

    # ---- Train vs Validation Loss ----
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(train_val_plot_path)
    plt.close()

    # ---- Residual Plot (Validation Set) ----
    all_preds, all_targets = [], []
    with torch.no_grad():
        for enc, dec, tgt in val_loader:
            enc, dec = enc.to(config.device), dec.to(config.device)
            output = model(enc, dec)
            all_preds.append(output.cpu())
            all_targets.append(tgt.cpu())

    preds = torch.cat(all_preds).numpy().flatten()
    targets = torch.cat(all_targets).numpy().flatten()
    residuals = targets - preds

    plt.figure(figsize=(6, 6))
    plt.scatter(targets, residuals, alpha=0.3)
    plt.axhline(0, linestyle='--')
    plt.xlabel("Actual abvaerk")
    plt.ylabel("Residual")
    plt.title("Residual Scatterplot (Validation)")
    plt.tight_layout()
    plt.savefig(residual_plot_path)
    plt.close()

    # ---- Test Predictions Plot ----
    all_preds, all_targets = [], []
    with torch.no_grad():
        for enc, dec, tgt in test_loader:
            enc, dec = enc.to(config.device), dec.to(config.device)
            output = model(enc, dec)
            all_preds.append(output.cpu())
            all_targets.append(tgt.cpu())

    preds = torch.cat(all_preds).numpy().flatten()
    targets = torch.cat(all_targets).numpy().flatten()

    plt.figure(figsize=(15, 5))
    plt.plot(targets, label="Actual abvaerk")
    plt.plot(preds, label="Predicted abvaerk")
    plt.xlabel("Time Steps (Test Set)")
    plt.ylabel("abvaerk")
    plt.title("Predicted vs Actual abvaerk (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(test_plot_path)
    plt.close()

    # ---- Per-Horizon Metrics Plot ----
    all_preds_h, all_targets_h = [], []
    with torch.no_grad():
        for enc, dec, tgt in test_loader:
            enc, dec = enc.to(config.device), dec.to(config.device)
            output = model(enc, dec)
            all_preds_h.append(output.cpu().numpy())
            all_targets_h.append(tgt.numpy())

    preds_h = np.concatenate(all_preds_h, axis=0)
    targets_h = np.concatenate(all_targets_h, axis=0)

    mse_per_horizon = np.mean((preds_h - targets_h) ** 2, axis=0)
    mae_per_horizon = np.mean(np.abs(preds_h - targets_h), axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(range(1, 169), mse_per_horizon, color='blue', linewidth=0.8)
    ax1.set_ylabel("MSE")
    ax1.set_title("Per-Horizon Forecast Error (Test Set)")
    ax1.axvline(x=24, color='gray', linestyle='--', alpha=0.5, label='24h')
    ax1.axvline(x=48, color='gray', linestyle=':', alpha=0.5, label='48h')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(1, 169), mae_per_horizon, color='red', linewidth=0.8)
    ax2.set_xlabel("Forecast Horizon (hours)")
    ax2.set_ylabel("MAE")
    ax2.axvline(x=24, color='gray', linestyle='--', alpha=0.5, label='24h')
    ax2.axvline(x=48, color='gray', linestyle='-', alpha=0.5, label='48h')
    ax2.axvline(x=72, color='gray', linestyle=':', alpha=0.5, label='72h')
    ax2.axvline(x=96, color='gray', linestyle='-.', alpha=0.5, label='96h')
    ax2.axvline(x=120, color='gray', linestyle=(0, (3, 1, 1, 1)), alpha=0.5, label='120h')
    ax2.axvline(x=144, color='gray', linestyle=(0, (5, 2)), alpha=0.5, label='144h')
    ax2.axvline(x=168, color='gray', linestyle=(0, (1, 1)), alpha=0.5, label='144h')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(horizon_plot_path)
    plt.close()

    print("Plots saved successfully.")


# Allow standalone execution: python3 LSTMTraining.py --local
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', help='Use local relative paths instead of server paths')
    args = parser.parse_args()
    main(local=args.local)
