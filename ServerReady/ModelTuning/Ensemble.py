from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import shutil
from LSTMModel import LSTMForecast, Config


# -----------------------------
# Paths relative to project root
# -----------------------------
project_root = Path(__file__).parent.parent.parent
dataset_path = project_root / "ServerReady" / "ModelTuning" / "dataset.pt"
ensemble_save_dir = project_root / "ServerReady" / "EnsembleModel" / "Models"
plot_path = project_root / "ServerReady" / "EnsembleModel" / "Plots" / "test_predictions_plot.png"

# -----------------------------
# Device and batch config
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
ensemble_size = 1

#This function runs the ensemble model that has inherent model uncertainty
def UncertaintyQuantification():
    #

    #Load dataset
    dataset = _DataLoader()

    trainLoader, valLoader, testLoader = _DatasetSplit(dataset)

    # Train Ensemble
    print("=== Training ensemble ===")
    model_paths = _TrainEnsemble(
        n_models=ensemble_size,
        train_loader=trainLoader,
        val_loader=valLoader,
        device=device,
        save_dir=ensemble_save_dir
    )
    print("Ensemble training complete.")

    # Load Ensemble Models
    models = _LoadEnsembleModels(ensemble_save_dir, device)
    print(f"Loaded {len(models)} ensemble models.")


    _EvaluateModel(testLoader, models, None, device)
    
#This part is all private functions
def _DataLoader():
    #

    # Create directories if they don't exist
    if ensemble_save_dir.exists():
        shutil.rmtree(ensemble_save_dir)  # delete folder and all contents
    ensemble_save_dir.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = torch.load(dataset_path)

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

    
    trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    testLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainLoader, valLoader, testLoader

def _EvaluateModel(test_loader, models, target_scaler, device):
    #

    def _Plot():
        #

        # Plot Predictions with Uncertainty
        plt.figure(figsize=(14,5))

        plt.plot(time_steps, targets_week, label="Actual abvaerk")
        plt.plot(time_steps, mean_preds_week, label="Predicted mean")

        plt.fill_between(
            time_steps,
            mean_preds_week - 2 * std_preds_week,
            mean_preds_week + 2 * std_preds_week,
            alpha=0.3,
            label="±2 std (uncertainty)"
        )

        plt.xlabel("Time Steps)")
        plt.ylabel("abvaerk")
        plt.title(f"Ensemble LSTM Forecast (One Week with Uncertainty for week {week_index+1})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        print(f"Plot saved to {plot_path}")
        print("Done.")

    # -----------------------------
    # Run Ensemble on Test Set
    # -----------------------------
    meanPredictions, stdPredictions, targets = [], [], []

    for enc, dec, tgt in test_loader:
        enc, dec = enc.to(device), dec.to(device)
        meanPredictions2, stdPredictions2 = _EnsemblePredict(models, enc, dec)

        meanPredictions.append(meanPredictions2)
        stdPredictions.append(stdPredictions2)
        targets.append(tgt)

    # Concatenate all batches
    meanPredictions = torch.cat(meanPredictions, dim=0).numpy().flatten()
    stdPredictions = torch.cat(stdPredictions, dim=0).numpy().flatten()
    targets = torch.cat(targets, dim=0).numpy().flatten()

    # -----------------------------
    # Select ONE WEEK of data
    # -----------------------------

    # choose which week (0 = first week)
    week_index = 1

    start = week_index * 168
    end = start + 168

    print(f"Selected week index: {week_index} -> time steps {start} to {end-1}")

    # safety check in case dataset is smaller
    end = min(end, len(meanPredictions))

    mean_preds_week = meanPredictions[start:end]
    std_preds_week = stdPredictions[start:end]
    targets_week = targets[start:end]

    #print(targets_week)

    time_steps = np.arange(len(mean_preds_week))

    #print(time_steps[0])

    print(targets_week[time_steps[0]])

    _Plot()

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
        config.epochs=1
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

UncertaintyQuantification()