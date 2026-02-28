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
    Train the LSTM model on the dataset and save the best checkpoint.

    Args:
        local: If True, use relative paths. If False, use server paths.
    """

    # -----------------------------
    # Paths
    # -----------------------------
    if local:
        _dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(_dir, "dataset.pt")
        model_dir = os.path.join(_dir, "..", "Models", "SingleLSTM")
        print("Running in LOCAL mode (relative paths)")
    else:
        dataset_path = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/dataset.pt"
        model_dir = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/Models/SingleLSTM"
        print("Running in SERVER mode (absolute paths)")

    os.makedirs(model_dir, exist_ok=True)

    # Auto-increment model version: model_v1.pth, model_v2.pth, ...
    existing = [f for f in os.listdir(model_dir) if f.startswith("model_v") and f.endswith(".pth")]
    existing_versions = []
    for f in existing:
        try:
            v = int(f.replace("model_v", "").replace(".pth", ""))
            existing_versions.append(v)
        except ValueError:
            pass
    next_version = max(existing_versions, default=0) + 1
    model_save_path = os.path.join(model_dir, f"model_v{next_version}.pth")
    print(f"Model will be saved as: model_v{next_version}.pth")

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

    config = Config()

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

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
                'train_losses': train_losses.copy(),
                'val_losses': val_losses.copy(),
            }, model_save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Save final loss curves into the checkpoint (includes epochs after best)
    checkpoint = torch.load(model_save_path)
    checkpoint['train_losses'] = train_losses
    checkpoint['val_losses'] = val_losses
    torch.save(checkpoint, model_save_path)

    print(f"Training complete. Best model saved at epoch {checkpoint['epoch']} "
          f"(val_loss={checkpoint['val_loss']:.4f}).")


# Allow standalone execution: python3 LSTMTraining.py --local
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', help='Use local relative paths instead of server paths')
    args = parser.parse_args()
    main(local=args.local)
