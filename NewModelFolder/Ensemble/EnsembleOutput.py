import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from pathlib import Path

from .EnsembleHelpers import _EnsemblePredict


def _EvaluateModel(test_loader, models, demand_mean, demand_std, device, plot_dir):
    """
    Runs ensemble evaluation with:
    - Mean prediction
    - Total uncertainty
    - Epistemic uncertainty
    - Aleatoric uncertainty

    Only targets are rescaled back to original units (MW),
    since predictions are already in MW.
    """

    meanPredictions = []
    stdPredictions = []
    epistemicPredictions = []
    aleatoricPredictions = []
    targets = []

    # -----------------------------
    # Run Ensemble on Test Set
    # -----------------------------
    for enc, dec, tgt in test_loader:
        enc, dec = enc.to(device), dec.to(device)

        mean_real, total_std_real, epistemic_real, aleatoric_real = \
            _EnsemblePredict(models, enc, dec, demand_mean, demand_std)

        meanPredictions.append(mean_real.cpu())
        stdPredictions.append(total_std_real.cpu())
        epistemicPredictions.append(epistemic_real.cpu())
        aleatoricPredictions.append(aleatoric_real.cpu())
        targets.append(tgt)

    # Concatenate batches
    meanPredictions = torch.cat(meanPredictions, dim=0).numpy().flatten()
    stdPredictions = torch.cat(stdPredictions, dim=0).numpy().flatten()
    epistemicPredictions = torch.cat(epistemicPredictions, dim=0).numpy().flatten()
    aleatoricPredictions = torch.cat(aleatoricPredictions, dim=0).numpy().flatten()
    targets = torch.cat(targets, dim=0).numpy().flatten()

    # -----------------------------
    # RESCALE ONLY TARGETS
    # -----------------------------
    targets_rescaled = targets * demand_std + demand_mean

    # -----------------------------
    # Select ONE WEEK of data
    # -----------------------------
    week_index = 1
    start = week_index * 168
    end = min(start + 168, len(meanPredictions))

    mean_preds_week = meanPredictions[start:end]
    std_preds_week = stdPredictions[start:end]
    targets_week = targets_rescaled[start:end]

    time_steps = np.arange(len(mean_preds_week))

    # -----------------------------
    # Generate Plots
    # -----------------------------
    _GeneratePlots(
        time_steps,
        targets_week,
        mean_preds_week,
        std_preds_week,
        week_index,
        meanPredictions,
        stdPredictions,
        targets_rescaled,
        epistemicPredictions,
        aleatoricPredictions,
        plot_dir
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
    targets,
    epistemicPredictions,
    aleatoricPredictions,
    plot_dir
):
    _PlotWeekForecast(time_steps, targets_week, mean_preds_week, std_preds_week, week_index, plot_dir)
    _PlotCalibration(meanPredictions, stdPredictions, targets, plot_dir)
    #_PlotResiduals(meanPredictions, targets, plot_dir)
    #_PlotUncertaintyHistogram(stdPredictions, plot_dir)
    #_PlotPredictionVsActual(meanPredictions, targets, plot_dir)
    #_PlotUncertaintyVsError(meanPredictions, stdPredictions, targets, plot_dir)


# ============================================================
# INDIVIDUAL PLOTS
# ============================================================

# Week forecast with uncertainty
def _PlotWeekForecast(time_steps, targets_week, mean_preds_week, std_preds_week, week_index, plot_dir):
    """
    Plots one-week forecast with uncertainty.
    Predictions and std are assumed to be already in MW.
    Targets are already rescaled before plotting.
    """
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

    plt.xlabel("Time Steps")
    plt.ylabel("abvaerk (MW)")
    plt.title(f"Ensemble LSTM Forecast (One Week with Uncertainty for week {week_index+1})")
    plt.legend()
    plt.tight_layout()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / f"week_forecast_ensemble_week{week_index+1}.png"
    
    plt.savefig(save_path)
    plt.close()

    print(f"Saved week forecast plot to {save_path}")


# Calibration Plot
def _PlotCalibration(meanPredictions, stdPredictions, targets, plot_dir):
    """
    Checks empirical coverage of prediction intervals.
    All inputs are in MW.
    """

    z_values = np.linspace(0.5, 3.0, 20)
    empirical_coverages = []

    for z in z_values:
        lower = meanPredictions - z * stdPredictions
        upper = meanPredictions + z * stdPredictions

        coverage = np.mean((targets >= lower) & (targets <= upper))
        empirical_coverages.append(coverage)

    theoretical_coverages = 2 * (0.5 * (1 + erf(z_values / np.sqrt(2)))) - 1

    plt.figure(figsize=(6,6))
    plt.plot(theoretical_coverages, empirical_coverages, marker="o")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("Theoretical Coverage")
    plt.ylabel("Empirical Coverage")
    plt.title("Calibration Curve")
    plt.tight_layout()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "calibration_curve.png"

    plt.savefig(save_path)
    plt.close()

    print(f"Saved calibration curve to {save_path}")
    print("stdPredictions min/max:", stdPredictions.min(), stdPredictions.max())


# --- Residuals over time ---
def _PlotResiduals(meanPredictions, targets, plot_dir):
    residuals = targets - meanPredictions

    plt.figure(figsize=(10,5))
    plt.plot(residuals)
    plt.title("Residuals over time")
    plt.xlabel("Time step")
    plt.ylabel("Error (target - prediction, MW)")
    plt.tight_layout()


    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "residuals_ensemble.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved residual plot to {save_path}")


# --- Histogram of predictive uncertainty ---
def _PlotUncertaintyHistogram(stdPredictions, plot_dir):
    plt.figure(figsize=(8,5))
    plt.hist(stdPredictions, bins=50)
    plt.title("Distribution of Predictive Uncertainty (std, MW)")
    plt.xlabel("Standard deviation (MW)")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "uncertainty_histogram.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved uncertainty histogram to {save_path}")


# --- Prediction vs Actual scatter ---
def _PlotPredictionVsActual(meanPredictions, targets, plot_dir):
    plt.figure(figsize=(6,6))
    plt.scatter(targets, meanPredictions, alpha=0.3)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], linestyle="--")
    plt.xlabel("Actual values (MW)")
    plt.ylabel("Predicted values (MW)")
    plt.title("Prediction vs Actual")
    plt.tight_layout()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "prediction_vs_actual.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved prediction vs actual plot to {save_path}")


# --- Uncertainty vs Error ---
def _PlotUncertaintyVsError(meanPredictions, stdPredictions, targets, plot_dir):
    errors = np.abs(targets - meanPredictions)

    plt.figure(figsize=(6,5))
    plt.scatter(stdPredictions, errors, alpha=0.3)
    plt.xlabel("Predicted uncertainty (std, MW)")
    plt.ylabel("Absolute error (MW)")
    plt.title("Uncertainty vs Prediction Error")
    plt.tight_layout()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "uncertainty_vs_error.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Saved uncertainty vs error plot to {save_path}")
