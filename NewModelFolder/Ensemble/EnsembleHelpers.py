from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
import shutil

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from LSTMModel import LSTMForecast, Config
from .EnsembleConfig import DATASET_PATH, ENSEMBLE_SAVE_DIR, BATCH_SIZE, EPOCHS


def _DataLoader():
    #

    # Create directories if they don't exist
    if ENSEMBLE_SAVE_DIR.exists():
        shutil.rmtree(ENSEMBLE_SAVE_DIR)  # delete folder and all contents
    ENSEMBLE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = torch.load(DATASET_PATH)

    #Create dataset
    return TensorDataset(
        dataset['encoder'],
        dataset['decoder'],
        dataset['target'])


def _DatasetSplit(dataset):
    # Split dataset into train/val/test

    val_ratio, test_ratio = 0.1, 0.1
    n_total = len(dataset)
    test_size = int(n_total * test_ratio)
    val_size = int(n_total * val_ratio)
    train_size = n_total - val_size - test_size

    train_dataset = Subset(dataset, range(0, train_size))
    val_dataset = Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(dataset, range(train_size + val_size, n_total))

    
    trainLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    testLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return trainLoader, valLoader, testLoader


def _EnsemblePredict(models, enc, dec):
    preds = []

    for model in models:
        model.eval()
        with torch.no_grad():
            pred = model(enc, dec)   # shape: (batch, horizon, 1)
            preds.append(pred.cpu())

    preds = torch.stack(preds, axis=0)  # (n_models, batch, horizon, 1)

    mean = preds.mean(axis=0)
    std  = preds.std(axis=0)

    return mean, std


def _LoadEnsembleModels(model_dir, device):
    model_dir = Path(model_dir)
    model_paths = sorted(model_dir.glob("lstm_seq2seq_model_*.pth"))

    models = []

    for path in model_paths:
        # allow Config to be unpickled safely
        with torch.serialization.safe_globals([Config]):
            checkpoint = torch.load(path, map_location=device, weights_only=False)

        config = checkpoint["config"]
        model = LSTMForecast(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)

    return models


def _TrainEnsemble(
    n_models,
    train_loader,
    val_loader,
    device,
    save_dir,
    base_seed=1000,
):
    """
    Train an ensemble of encoder-decoder LSTM models with different random seeds.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    model_paths = []

    for i in range(n_models):
        seed = base_seed + i
        print(f"\n=== Training ensemble model {i+1}/{n_models} (seed={seed}) ===")

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize model and config
        config = Config()
        config.epochs=EPOCHS
        model = LSTMForecast(config).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0
        patience = 50

        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_loss = 0

            for enc, dec, tgt in tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}"):
                enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)

                optimizer.zero_grad()
                output = model(enc, dec)
                loss = criterion(output, tgt)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * enc.size(0)

            train_loss = epoch_loss / len(train_loader.dataset)

            # Validation
            model.eval()
            val_loss_epoch = 0
            with torch.no_grad():
                for enc, dec, tgt in val_loader:
                    enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
                    output = model(enc, dec)
                    val_loss_epoch += criterion(output, tgt).item() * enc.size(0)

            val_loss = val_loss_epoch / len(val_loader.dataset)
            scheduler.step(val_loss)

            print(f"Model {i+1} | Epoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                model_path = save_dir / f"lstm_seq2seq_model_{i+1}.pth"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": config
                }, model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"Saved ensemble model to {model_path}")
        model_paths.append(model_path)

    return model_paths
