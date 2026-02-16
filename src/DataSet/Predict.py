import json
import pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from LSTM import LSTMModel

# Paths relative to this script
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / 'lstm_model.pth'
DATASET_PATH = SCRIPT_DIR / 'dataset.json'


def load_model():
    """Load the trained LSTM model."""
    # Newer PyTorch versions restrict which globals can be unpickled by
    # default. The saved checkpoint contains sklearn's StandardScaler which
    # needs to be allowlisted for safe unpickling. We try to use the
    # safe_globals context manager and fall back to a direct load if that
    # fails.
    try:
        import sklearn.preprocessing._data as skdata
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

    print(f"Model loaded from {MODEL_PATH}")
    print(f"Input size: {input_size}")

    return model, feature_scaler, target_scaler


def load_dataset():
    """Load the dataset to get sample data for prediction."""
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)

    samples = np.array(data['samples'], dtype=np.float32)
    targets = np.array(data['targets'], dtype=np.float32)

    print(f"Dataset loaded: {samples.shape[0]} samples")
    print(f"Features: {data['feature_names']}")

    return samples, targets, data


def predict_single_sample(model, sample, feature_scaler, target_scaler):
    """
    Make a prediction for a single sample (168 hour forecast).

    Args:
        model: Trained LSTM model
        sample: Input sample of shape (168, n_features)
        feature_scaler: Scaler for input features
        target_scaler: Scaler for output target

    Returns:
        Predicted energy usage value
    """
    # Normalize the input
    n_features = sample.shape[1]
    sample_flat = sample.reshape(-1, n_features)
    sample_scaled = feature_scaler.transform(sample_flat).reshape(1, 168, n_features)

    # Convert to tensor
    sample_tensor = torch.FloatTensor(sample_scaled)

    # Make prediction
    with torch.no_grad():
        prediction_scaled = model(sample_tensor).numpy().flatten()[0]

    # Inverse transform to get original scale
    prediction = target_scaler.inverse_transform([[prediction_scaled]])[0][0]

    return prediction


def predict_week_ahead(model, feature_scaler, target_scaler, samples, targets, start_idx=None):
    """
    Predict energy demand for one week (168 hours) into the future.

    Uses the last available sample as the starting point, or a specified index.

    Args:
        model: Trained LSTM model
        feature_scaler: Scaler for input features
        target_scaler: Scaler for output target
        samples: All samples from dataset
        targets: All targets from dataset
        start_idx: Index to start prediction from (default: last sample)

    Returns:
        Dictionary with predictions and actuals
    """
    if start_idx is None:
        start_idx = len(samples) - 168  # Start 168 hours before the end

    print(f"\nPredicting from sample index {start_idx}")

    predictions = []
    actuals = []

    # Predict for each hour in the week
    for i in range(168):
        idx = start_idx + i
        if idx >= len(samples):
            print(f"Warning: Reached end of dataset at hour {i}")
            break

        sample = samples[idx]
        actual = targets[idx]

        prediction = predict_single_sample(model, sample, feature_scaler, target_scaler)

        predictions.append(prediction)
        actuals.append(actual)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    print(f"\nWeek-ahead Prediction Metrics:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    return {
        'predictions': predictions,
        'actuals': actuals,
        'metrics': {'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape},
        'start_idx': start_idx
    }


def plot_week_prediction(results, save_path=None):
    """Plot the week-ahead prediction vs actual values."""
    predictions = results['predictions']
    actuals = results['actuals']
    hours = np.arange(len(predictions))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Full week comparison
    ax1 = axes[0, 0]
    ax1.plot(hours, actuals, 'b-', label='Actual', linewidth=1.5, alpha=0.8)
    ax1.plot(hours, predictions, 'r-', label='Predicted', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Hours Ahead', fontsize=12)
    ax1.set_ylabel('Energy Usage (MWh)', fontsize=12)
    ax1.set_title('Week-Ahead Prediction (168 hours)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add day markers
    for day in range(1, 8):
        ax1.axvline(x=day*24, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Daily breakdown (by day of week)
    ax2 = axes[0, 1]
    for day in range(7):
        start = day * 24
        end = start + 24
        if end <= len(predictions):
            day_hours = np.arange(24)
            ax2.plot(day_hours, actuals[start:end], 'b-', alpha=0.3 + day*0.1)
            ax2.plot(day_hours, predictions[start:end], 'r--', alpha=0.3 + day*0.1)
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.set_ylabel('Energy Usage (MWh)', fontsize=12)
    ax2.set_title('Daily Patterns (Blue=Actual, Red=Predicted)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Prediction error over time
    ax3 = axes[1, 0]
    errors = predictions - actuals
    ax3.bar(hours, errors, color=['green' if e >= 0 else 'red' for e in errors], alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Hours Ahead', fontsize=12)
    ax3.set_ylabel('Prediction Error (MWh)', fontsize=12)
    ax3.set_title('Prediction Error Over Time', fontsize=14)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Scatter plot with perfect prediction line
    ax4 = axes[1, 1]
    ax4.scatter(actuals, predictions, alpha=0.6, s=30)
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax4.set_xlabel('Actual Energy Usage (MWh)', fontsize=12)
    ax4.set_ylabel('Predicted Energy Usage (MWh)', fontsize=12)
    ax4.set_title('Predicted vs Actual', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPrediction plot saved to {save_path}")

    plt.show()


def plot_hourly_errors(results, save_path=None):
    """Plot error analysis by hour of day."""
    predictions = results['predictions']
    actuals = results['actuals']

    # Calculate errors by hour of day
    hourly_errors = {h: [] for h in range(24)}
    hourly_mape = {h: [] for h in range(24)}

    for i, (pred, actual) in enumerate(zip(predictions, actuals)):
        hour = i % 24
        hourly_errors[hour].append(abs(pred - actual))
        hourly_mape[hour].append(abs((pred - actual) / actual) * 100)

    # Average errors per hour
    avg_errors = [np.mean(hourly_errors[h]) for h in range(24)]
    avg_mape = [np.mean(hourly_mape[h]) for h in range(24)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MAE by hour
    ax1 = axes[0]
    ax1.bar(range(24), avg_errors, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error (MWh)', fontsize=12)
    ax1.set_title('Prediction Error by Hour of Day', fontsize=14)
    ax1.set_xticks(range(24))
    ax1.grid(True, alpha=0.3, axis='y')

    # MAPE by hour
    ax2 = axes[1]
    ax2.bar(range(24), avg_mape, color='coral', alpha=0.7)
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.set_ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
    ax2.set_title('MAPE by Hour of Day', fontsize=14)
    ax2.set_xticks(range(24))
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Hourly errors plot saved to {save_path}")

    plt.show()


def main():
    print("=" * 60)
    print("Week-Ahead Energy Demand Prediction")
    print("=" * 60)

    # Load model and data
    model, feature_scaler, target_scaler = load_model()
    samples, targets, metadata = load_dataset()

    # Make week-ahead prediction starting from a random point
    # Use a point that has 168 hours of data after it
    np.random.seed(42)
    max_start = len(samples) - 168
    start_idx = np.random.randint(0, max_start)

    results = predict_week_ahead(
        model, feature_scaler, target_scaler,
        samples, targets, start_idx=start_idx
    )

    # Create plots directory
    plots_dir = SCRIPT_DIR / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Generate plots
    plot_week_prediction(results, save_path=plots_dir / 'week_prediction.png')
    plot_hourly_errors(results, save_path=plots_dir / 'hourly_errors.png')

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Predicted {len(results['predictions'])} hours (1 week)")
    print(f"Starting from sample index: {results['start_idx']}")
    print(f"\nMetrics:")
    for key, value in results['metrics'].items():
        if key == 'mape':
            print(f"  {key.upper()}: {value:.2f}%")
        else:
            print(f"  {key.upper()}: {value:.4f}")


if __name__ == '__main__':
    main()

