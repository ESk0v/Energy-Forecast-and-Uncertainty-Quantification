import pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from LSTMModel import LSTMModel
from Data      import ForecastDataset, load_dataset, normalize_data
from Training  import train_model, evaluate_model
from Plotting  import (
    plot_training_curves,
    plot_predictions_vs_actuals,
    plot_residuals,
    plot_error_distribution,
)

# =============================================================================
# CONFIG — change values here, no need to dig into the code
# =============================================================================

# --- Paths ---
SCRIPT_DIR   = pathlib.Path(__file__).resolve().parent.parent
DATASET_PATH = SCRIPT_DIR / 'Files' / 'dataset.json'
MODEL_PATH   = SCRIPT_DIR / 'Files' / 'LSTMModels' / 'lstm_model.pth'
PLOTS_DIR    = SCRIPT_DIR / 'Files' / 'TrainingPlots'

# --- Reproducibility ---
RANDOM_SEED  = 618

# --- Data ---
NORMALIZE    = True
TEST_SIZE    = 0.30   # fraction held out from full dataset (val + test)
VAL_SPLIT    = 0.50   # fraction of the above that becomes val (rest is test)
                      # result: 70% train | 15% val | 15% test
# --- Training ---
BATCH_SIZE   = 8
EPOCHS       = 50000
LEARNING_RATE = 0.002

# --- Model ---
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
DROPOUT      = 0.2

# --- Output toggles ---
SAVE_MODEL   = True
SAVE_PLOTS   = True
SHOW_PLOTS   = False

# =============================================================================


def main():
    # --- Reproducibility ---
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load data ---
    samples, targets, metadata = load_dataset(DATASET_PATH)

    # --- Split: train / val / test ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        samples, targets, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=VAL_SPLIT, random_state=RANDOM_SEED
    )

    print(f"\nData split:")
    print(f"  Train : {len(X_train)} samples")
    print(f"  Val   : {len(X_val)} samples")
    print(f"  Test  : {len(X_test)} samples")

    # --- Normalize ---
    if NORMALIZE:
        X_train, X_val, y_train, y_val, feature_scaler, target_scaler = normalize_data(
            X_train, X_val, y_train, y_val
        )
        X_test = feature_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        y_test = target_scaler.transform(y_test.reshape(-1, 1)).flatten()
    else:
        feature_scaler = None
        target_scaler  = None

    # --- Data loaders ---
    train_loader = DataLoader(ForecastDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ForecastDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(ForecastDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)

    # --- Model ---
    input_size = metadata['n_features']
    model = LSTMModel(
        input_size  = input_size,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        dropout     = DROPOUT,
    )
    print(f"\nModel architecture:\n{model}")

    # --- Train ---
    print("\nTraining...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        epochs = EPOCHS,
        lr     = LEARNING_RATE,
        device = device,
    )

    # --- Evaluate ---
    predictions, actuals, metrics = evaluate_model(model, test_loader, target_scaler, device)

    # --- Plots ---
    plots_dir = PLOTS_DIR
    plots_dir.mkdir(exist_ok=True)

    save = lambda filename: plots_dir / filename if SAVE_PLOTS else None

    plot_training_curves(train_losses, val_losses,              save_path=save('training_curves.png'), show_plots=SHOW_PLOTS)
    plot_predictions_vs_actuals(predictions, actuals,           save_path=save('predictions_vs_actuals.png'), show_plots=SHOW_PLOTS)
    plot_residuals(predictions, actuals,                        save_path=save('residuals.png'), show_plots=SHOW_PLOTS)
    plot_error_distribution(predictions, actuals,               save_path=save('error_distribution.png'), show_plots=SHOW_PLOTS)

    # --- Save model ---
    if SAVE_MODEL:
        torch.save({
            'model_state_dict' : model.state_dict(),
            'feature_scaler'   : feature_scaler,
            'target_scaler'    : target_scaler,
            'metrics'          : metrics,
            'input_size'       : input_size,
        }, MODEL_PATH)
        print(f"\nModel saved to {MODEL_PATH}")


if __name__ == '__main__':
    main()