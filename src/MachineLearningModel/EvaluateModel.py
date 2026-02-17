import os
import pathlib
from matplotlib import pyplot as plt
import torch
import numpy as np
import sklearn.preprocessing._data as skdata

from LSTMModel import LSTMModel
from Data import load_dataset

# Paths relative to this script

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = SCRIPT_DIR / 'Files' / 'LSTMModels' / 'lstm_model.pth'
DATASET_PATH = SCRIPT_DIR / 'Files' / 'dataset.json'

def ResidualsPlot(save_path=None, show_plots=True):

    def PlotAllHorizonsResiduals(preds_list, acts_list, horizons, save_path=None, show_plots=True):
        
        def compute_mape(actuals, predictions):
            return np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
        n_horizons = len(horizons)
        fig, axes = plt.subplots(1, n_horizons, figsize=(6 * n_horizons, 4), sharey=True)

        if n_horizons == 1:
            axes = [axes]

        colors = ['blue', 'green', 'red']  # match Predicted vs Actual

        for i, horizon in enumerate(horizons):
            predictions = preds_list[i]
            actuals = acts_list[i]
            residuals = predictions - actuals
            mape = compute_mape(actuals, predictions)

            ax = axes[i]
            # --- Scatter plot for residuals ---
            ax.scatter(np.arange(len(residuals)), residuals, s=4, alpha=0.2, color=colors[i])
            ax.axhline(0, color='k', linestyle='--', linewidth=0.8)  # zero line
            ax.set_title(f'{horizon} Horizon Residuals', fontsize=14)
            ax.set_xlabel('Sample Index', fontsize=12)
            if i == 0:
                ax.set_ylabel('Residual (MWh)', fontsize=12)
            ax.legend([f"Residuals\nMAPE: {mape:.2f}%"], fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Residuals over time plot saved to {save_path}")
        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    # --- Load model + scalers ---
    model, feature_scaler, target_scaler = _LoadModel(False)

    # --- Load dataset ---
    samples, targets, _ = load_dataset(DATASET_PATH, False)

    # --- Evaluate horizons ---
    pred_1h, act_1h = _EvaluateHorizon(model, samples, targets, feature_scaler, target_scaler, horizon=0)
    pred_24h, act_24h = _EvaluateHorizon(model, samples, targets, feature_scaler, target_scaler, horizon=23)
    pred_168h, act_168h = _EvaluateHorizon(model, samples, targets, feature_scaler, target_scaler, horizon=167)

    # --- Plot residuals ---
    PlotAllHorizonsResiduals(
        preds_list=[pred_1h, pred_24h, pred_168h],
        acts_list=[act_1h, act_24h, act_168h],
        horizons=['1h', '24h', '168h'],
        save_path=save_path,
        show_plots=show_plots
    )

def PredictionsVsActualsPlot(save_path=None, show_plots=True):

    def PlotAllHorizons(preds_list, acts_list, horizons, save_path=True, show_plots=True):

        def compute_mape(actuals, predictions):
            return np.mean(np.abs((actuals - predictions) / actuals)) * 100

        colors = ['blue', 'green', 'red']
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))

        for i, horizon in enumerate(horizons):
            actuals = acts_list[i]
            predictions = preds_list[i]
            mape = compute_mape(actuals, predictions)

            # --- Scatter plot ---
            ax1 = axes[0, i]
            ax1.scatter(actuals, predictions, alpha=0.5, s=10, color=colors[i])
            min_val = min(actuals.min(), predictions.min())
            max_val = max(actuals.max(), predictions.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')  # black dashed line
            ax1.set_xlabel('Actual Energy Usage (MWh)', fontsize=12)
            ax1.set_ylabel('Predicted Energy Usage (MWh)', fontsize=12)
            ax1.set_title(f'{horizon} Horizon: Predicted vs Actual', fontsize=14)
            ax1.legend(title=f"MAPE: {mape:.2f}%", fontsize=10)
            ax1.grid(True, alpha=0.3)

            # --- Line plot (full series) ---
            ax2 = axes[1, i]
            ax2.plot(actuals, 'b-', label='Actual', linewidth=0.09, alpha=0.9)        # thinner + more transparent
            ax2.plot(predictions, 'r-', label='Predicted', linewidth=0.09, alpha=1)  # thinner + more transparent
            ax2.set_xlabel('Sample Index', fontsize=12)
            ax2.set_ylabel('Energy Usage (MWh)', fontsize=12)
            ax2.set_title(f'{horizon} Horizon: Full Series', fontsize=14)
            ax2.legend(title=f"MAPE: {mape:.2f}%", fontsize=10)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        _SavePlot(save_path, show_plots)
    
    # Load model + scalers
    model, feature_scaler, target_scaler = _LoadModel(False)

    # Load dataset
    samples, targets, _ = load_dataset(DATASET_PATH, False)

    # Evaluate horizons
    pred_1h, act_1h = _EvaluateHorizon(model, samples, targets, feature_scaler, target_scaler, horizon=0)
    pred_24h, act_24h = _EvaluateHorizon(model, samples, targets, feature_scaler, target_scaler, horizon=23)
    pred_168h, act_168h = _EvaluateHorizon(model, samples, targets, feature_scaler, target_scaler, horizon=167)
    
    PlotAllHorizons(
        preds_list=[pred_1h, pred_24h, pred_168h],
        acts_list=[act_1h, act_24h, act_168h],
        horizons=['1h', '24h', '168h'],
        save_path=save_path, 
        show_plots=show_plots)

def TrainingValidationPlot(train_losses, val_losses, save_path=None, show_plots=True):
    """
    Plot training vs validation losses (or any metric) over epochs.

    Args:
        train_losses : list or np.ndarray of training losses
        val_losses   : list or np.ndarray of validation losses
        save_path    : str or Path to save figure (optional)
        show_plots   : bool, whether to show the plot
    """
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot training vs validation
    ax.plot(epochs, train_losses, 'b-', label='Training', linewidth=1.5, alpha=0.9)
    ax.plot(epochs, val_losses,   'r-', label='Validation', linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training vs Validation Loss', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Training vs Validation plot saved to {save_path}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

def FirstWeekPredictionPlot(save_path=None, show_plots=True):

    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    # --- Load ---
    model, feature_scaler, target_scaler = _LoadModel()
    model.eval()

    samples, targets, _ = load_dataset(DATASET_PATH, False)

    predictions = []
    actual_week = []

    # Build first 168-hour forecast sequentially
    for i in range(168):

        sample_i = samples[i]      # (168, 9)
        target_i = targets[i]      # scalar

        sample_scaled = feature_scaler.transform(sample_i)
        sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_scaled = model(sample_tensor).squeeze().item()

        pred = target_scaler.inverse_transform([[pred_scaled]])[0, 0]

        predictions.append(pred)
        actual_week.append(target_i)

    predictions = np.array(predictions)
    actual_week = np.array(actual_week)

    hours = np.arange(168)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(hours, actual_week, label='Actual', linewidth=1.5, alpha=0.9)
    ax.plot(hours, predictions, label='Predicted', linewidth=1.5, alpha=0.9)

    ax.set_xlabel('Hour')
    ax.set_ylabel('Energy Usage (MWh)')
    ax.set_title('First 168 Hour Forecast')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

def _LoadModel(verbose=True):
    """Load the trained LSTM model."""
    # Newer PyTorch versions restrict which globals can be unpickled by
    # default. The saved checkpoint contains sklearn's StandardScaler which
    # needs to be allowlisted for safe unpickling. We try to use the
    # safe_globals context manager and fall back to a direct load if that
    # fails.
    try:
        with torch.serialization.safe_globals([skdata.StandardScaler]):
            checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    except Exception:
        # Fallback: try loading without the safe_globals wrapper. This may
        # be less secure, but will work for trusted local checkpoints.
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

    input_size = checkpoint['input_size']
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    feature_scaler = checkpoint['feature_scaler']
    target_scaler = checkpoint['target_scaler']
    if verbose:
        print(f"Model loaded from {MODEL_PATH}")
        print(f"Input size: {input_size}")

    return model, feature_scaler, target_scaler

def _EvaluateHorizon(model, samples, targets, feature_scaler, target_scaler, horizon=0):

        predictions = []
        actuals = []

        max_idx = len(samples) - horizon

        model.eval()  # important

        for i in range(max_idx):

            sample = samples[i]
            actual = targets[i + horizon]

            # --- Normalize input ---
            n_features = sample.shape[1]
            sample_flat = sample.reshape(-1, n_features)
            sample_scaled = feature_scaler.transform(sample_flat)
            sample_scaled = sample_scaled.reshape(1, 168, n_features)

            sample_tensor = torch.FloatTensor(sample_scaled)

            # --- Predict ---
            with torch.no_grad():
                prediction_scaled = model(sample_tensor).numpy().flatten()[0]

            prediction = target_scaler.inverse_transform(
                [[prediction_scaled]]
            )[0][0]

            predictions.append(prediction)
            actuals.append(actual)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        return predictions, actuals

def _SavePlot(save_path, show_plots):
    """Save the current figure if a path is given, then display it."""
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show_plots: plt.show()