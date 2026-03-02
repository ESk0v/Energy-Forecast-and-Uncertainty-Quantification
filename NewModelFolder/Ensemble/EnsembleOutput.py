import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .EnsembleHelpers import _EnsemblePredict
from .EnsembleConfig import PLOT_DIR


def _EvaluateModel(test_loader, models, device):
    #

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
    week_index = 1

    start = week_index * 168
    end = start + 168

    # safety check in case dataset is smaller
    end = min(end, len(meanPredictions))

    mean_preds_week = meanPredictions[start:end]
    std_preds_week = stdPredictions[start:end]
    targets_week = targets[start:end]

    time_steps = np.arange(len(mean_preds_week))

    print(targets_week[time_steps[0]])

    # Call all plots from one place
    _GeneratePlots(
        time_steps,
        targets_week,
        mean_preds_week,
        std_preds_week,
        week_index,
        meanPredictions,
        stdPredictions,
        targets
    )


# ============================================================
# MASTER PLOT FUNCTION
# ============================================================

def _GeneratePlots(
    time_steps,
    targets_week,
    mean_preds_week,
    std_preds_week,
    week_index,
    meanPredictions,
    stdPredictions,
    targets
):
    _PlotWeekForecast(
        time_steps,
        targets_week,
        mean_preds_week,
        std_preds_week,
        week_index
    )

    #_PlotResiduals(meanPredictions, targets)

    #_PlotUncertaintyHistogram(stdPredictions)

    #_PlotPredictionVsActual(meanPredictions, targets)

    #_PlotUncertaintyVsError(meanPredictions, stdPredictions, targets)


# ============================================================
# INDIVIDUAL PLOTS
# ============================================================

def _PlotWeekForecast(time_steps, targets_week, mean_preds_week, std_preds_week, week_index):
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

    save_path = PLOT_DIR / "week_forecast_ensemble.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved week forecast plot to {save_path}")


# --- Residuals over time ---
def _PlotResiduals(meanPredictions, targets):
    residuals = targets - meanPredictions

    plt.figure(figsize=(10,5))
    plt.plot(residuals)
    plt.title("Residuals over time")
    plt.xlabel("Time step")
    plt.ylabel("Error (target - prediction)")
    plt.tight_layout()

    save_path = PLOT_DIR / "residuals_ensemble.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved residual plot to {save_path}")


# --- Histogram of predictive uncertainty ---
def _PlotUncertaintyHistogram(stdPredictions):
    plt.figure(figsize=(8,5))
    plt.hist(stdPredictions, bins=50)
    plt.title("Distribution of Predictive Uncertainty (std)")
    plt.xlabel("Standard deviation")
    plt.ylabel("Frequency")
    plt.tight_layout()

    save_path = PLOT_DIR / "uncertainty_histogram.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved uncertainty histogram to {save_path}")


# --- Prediction vs Actual scatter ---
def _PlotPredictionVsActual(meanPredictions, targets):
    plt.figure(figsize=(6,6))
    plt.scatter(targets, meanPredictions, alpha=0.3)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], linestyle="--")
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.title("Prediction vs Actual")
    plt.tight_layout()

    save_path = PLOT_DIR / "prediction_vs_actual.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved prediction vs actual plot to {save_path}")


# --- Uncertainty vs Error ---
def _PlotUncertaintyVsError(meanPredictions, stdPredictions, targets):
    errors = np.abs(targets - meanPredictions)

    plt.figure(figsize=(6,5))
    plt.scatter(stdPredictions, errors, alpha=0.3)
    plt.xlabel("Predicted uncertainty (std)")
    plt.ylabel("Absolute error")
    plt.title("Uncertainty vs Prediction Error")
    plt.tight_layout()

    save_path = PLOT_DIR / "uncertainty_vs_error.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved uncertainty vs error plot to {save_path}")