import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(train_losses, val_losses, save_path=None, show_plots=True):
    """
    Plot training and validation loss curves over epochs.

    Args:
        train_losses : list of training loss values per epoch
        val_losses   : list of validation loss values per epoch
        save_path    : optional path to save the figure (e.g. plots/training_curves.png)
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss',   linewidth=2)
    plt.plot(epochs, val_losses,   'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch',      fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_and_show(save_path, show_plots)


def plot_predictions_vs_actuals(predictions, actuals, save_path=None, show_plots=True):
    """
    Two-panel plot: scatter of predicted vs actual, and a time series overlay.

    Args:
        predictions : np.ndarray of predicted values in original scale
        actuals     : np.ndarray of actual values in original scale
        save_path   : optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Scatter plot ---
    ax1.scatter(actuals, predictions, alpha=0.5, s=10)
    min_val, max_val = min(actuals.min(), predictions.min()), max(actuals.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Energy Usage (MWh)',    fontsize=12)
    ax1.set_ylabel('Predicted Energy Usage (MWh)', fontsize=12)
    ax1.set_title('Predicted vs Actual Values', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Time series overlay (first 200 samples) ---
    n_show = min(200, len(predictions))
    ax2.plot(actuals[:n_show],     'b-', label='Actual',    linewidth=1.5, alpha=0.8)
    ax2.plot(predictions[:n_show], 'r-', label='Predicted', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Sample Index',          fontsize=12)
    ax2.set_ylabel('Energy Usage (MWh)',    fontsize=12)
    ax2.set_title(f'Actual vs Predicted (First {n_show} Samples)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_and_show(save_path, show_plots)


def plot_residuals(predictions, actuals, save_path=None, show_plots=True):
    """
    Three-panel residual analysis: residuals vs predicted, histogram, and sorted residuals.

    Args:
        predictions : np.ndarray of predicted values in original scale
        actuals     : np.ndarray of actual values in original scale
        save_path   : optional path to save the figure
    """
    residuals = predictions - actuals

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # --- Residuals vs predicted ---
    ax1.scatter(predictions, residuals, alpha=0.5, s=10)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Values', fontsize=12)
    ax1.set_ylabel('Residuals',        fontsize=12)
    ax1.set_title('Residuals vs Predicted', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # --- Residual histogram ---
    ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residual Value', fontsize=12)
    ax2.set_ylabel('Frequency',      fontsize=12)
    ax2.set_title('Residual Distribution', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # --- Sorted residuals (Q-Q style) ---
    sorted_residuals      = np.sort(residuals)
    theoretical_quantiles = np.linspace(0, 1, len(sorted_residuals))
    ax3.plot(theoretical_quantiles, sorted_residuals, 'b-', linewidth=1)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Quantile',       fontsize=12)
    ax3.set_ylabel('Residual Value', fontsize=12)
    ax3.set_title('Sorted Residuals', fontsize=14)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_and_show(save_path, show_plots)


def plot_error_distribution(predictions, actuals, save_path=None, show_plots=True):
    """
    Two-panel error distribution: absolute errors and percentage errors (clipped at 50%).

    Args:
        predictions : np.ndarray of predicted values in original scale
        actuals     : np.ndarray of actual values in original scale
        save_path   : optional path to save the figure
    """
    errors            = np.abs(predictions - actuals)
    percentage_errors = np.abs((predictions - actuals) / actuals) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Absolute error histogram ---
    ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=np.mean(errors),   color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    ax1.axvline(x=np.median(errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f}')
    ax1.set_xlabel('Absolute Error (MWh)', fontsize=12)
    ax1.set_ylabel('Frequency',            fontsize=12)
    ax1.set_title('Absolute Error Distribution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Percentage error histogram (clipped at 50%) ---
    clipped_pct_errors = np.clip(percentage_errors, 0, 50)
    ax2.hist(clipped_pct_errors, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax2.axvline(x=np.mean(percentage_errors),   color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(percentage_errors):.1f}%')
    ax2.axvline(x=np.median(percentage_errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(percentage_errors):.1f}%')
    ax2.set_xlabel('Percentage Error (%)', fontsize=12)
    ax2.set_ylabel('Frequency',            fontsize=12)
    ax2.set_title('Percentage Error Distribution (clipped at 50%)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_and_show(save_path, show_plots)


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _save_and_show(save_path, show_plots):
    """Save the current figure if a path is given, then display it."""
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    if show_plots: plt.show()