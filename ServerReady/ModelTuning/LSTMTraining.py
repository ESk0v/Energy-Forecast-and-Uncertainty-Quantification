import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from tqdm import tqdm
from LSTMModel import Config, LSTMForecast

# -----------------------------
# [CHANGE] --local flag: use relative paths when running locally for testing.
# Usage: python3 LSTMTraining.py --local
# Without the flag, server paths are used (default behavior).
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--local', action='store_true', help='Use local relative paths instead of server paths')
args = parser.parse_args()

if args.local:
    # Relative to the script's directory (ServerReady/ModelTuning/)
    _dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(_dir, "dataset.pt")
    model_save_path = os.path.join(_dir, "best_lstm_forecast_model.pth")
    train_val_plot_path = os.path.join(_dir, "train_val_loss.png")
    residual_plot_path = os.path.join(_dir, "residuals.png")
    test_plot_path = os.path.join(_dir, "test_predictions.png")
    horizon_plot_path = os.path.join(_dir, "per_horizon_metrics.png")
    print("Running in LOCAL mode (relative paths)")
else:
    # Server paths (default)
    dataset_path = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/dataset.pt"
    model_save_path = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/best_lstm_forecast_model.pth"
    train_val_plot_path = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/train_val_loss.png"
    residual_plot_path = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/residuals.png"
    test_plot_path = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/test_predictions.png"
    # [CHANGE] Added per-horizon metrics plot to see how accuracy degrades with forecast distance
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
# Train/Val/Test Split
# -----------------------------
val_ratio, test_ratio = 0.1, 0.1
val_size = int(len(full_dataset) * val_ratio)
test_size = int(len(full_dataset) * test_ratio)
train_size = len(full_dataset) - val_size - test_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

config = Config()  # create instance

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# -----------------------------
# Model, Loss, Optimizer
# -----------------------------
model = LSTMForecast(config).to(config.device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

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

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), model_save_path)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

# -----------------------------
# Load best model
# -----------------------------
model.load_state_dict(torch.load(model_save_path))
model.eval()
print("Training complete. Best model loaded.")

# -----------------------------
# PLOTS
# -----------------------------
print("Generating plots...")

# ---- Train vs Validation Loss ----
plt.figure(figsize=(8,5))
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

plt.figure(figsize=(6,6))
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

plt.figure(figsize=(15,5))
plt.plot(targets, label="Actual abvaerk")
plt.plot(preds, label="Predicted abvaerk")
plt.xlabel("Time Steps (Test Set)")
plt.ylabel("abvaerk")
plt.title("Predicted vs Actual abvaerk (Test Set)")
plt.legend()
plt.tight_layout()
plt.savefig(test_plot_path)
plt.close()

print("Plots saved successfully.")