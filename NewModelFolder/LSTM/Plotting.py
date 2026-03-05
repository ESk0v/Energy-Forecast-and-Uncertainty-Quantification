"""
Plotting — Evaluation plots for the trained LSTM model.
=========================================================
Loads the saved checkpoint and dataset, rebuilds the model, runs inference
on the validation and test sets, and generates all evaluation plots.

Can be run independently after training is complete:
    python3 Plotting.py --local
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

# LSTMModel.py lives in the parent directory (NewModelFolder/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from LSTMModel import Config, LSTMForecast
from LSTM.GenerateREADME import generate_evaluation_readme

# Dataset starts at 2023-01-01 01:00, one row per hour, no gaps
DATASET_START = pd.Timestamp("2023-01-01 01:00")
ENCODER_HISTORY = 168  # must match DatasetCreation.py


def main(local=False, filePaths=None):
    """
    Generate all evaluation plots from a trained model checkpoint.

    Args:
        local:     If True, use relative paths (standalone fallback).
        filePaths: List of [dataset_path, model_dir, plot_dir].
                   When called from Main.py this is always provided.
                   When run standalone (--local) it is derived here.
    """

    # -----------------------------
    # Paths
    # -----------------------------
    if filePaths is not None:
        # Propagated from LSTMMain.py — filePaths[2] is the specific run folder
        dataset_path = filePaths[0]
        model_dir    = filePaths[1]
        run_dir      = filePaths[2]
    else:
        # Standalone fallback (python3 Plotting.py --local):
        # scan SingleLSTM/ for the highest-versioned run folder
        base_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        dataset_path = os.path.join(base_dir, "Files", "dataset.pt")
        model_dir    = os.path.join(base_dir, "Models", "SingleLSTM")
        existing     = [f for f in os.listdir(model_dir)
                        if os.path.isdir(os.path.join(model_dir, f))
                        and f.startswith("model_v")]
        versions = []
        for f in existing:
            try:
                versions.append(int(f.replace("model_v", "")))
            except ValueError:
                pass
        if not versions:
            raise FileNotFoundError(f"No versioned run folders found in {model_dir}")
        run_dir = os.path.join(model_dir, f"model_v{max(versions)}")

    plot_dir = run_dir  # plots, READMEs, and model.pth all live together
    os.makedirs(plot_dir, exist_ok=True)

    model_save_path = os.path.join(run_dir, "model.pth")
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"model.pth not found in run folder: {run_dir}")

    print(f"Plotting — dataset : {dataset_path}")
    print(f"Plotting — run dir : {run_dir}")

    train_val_plot_path  = os.path.join(plot_dir, "train_val_loss.png")
    test_plot_path       = os.path.join(plot_dir, "test_predictions.png")
    scatter_plot_path    = os.path.join(plot_dir, "actual_vs_predicted.png")
    residuals_plot_path  = os.path.join(plot_dir, "residuals.png")
    horizon_plot_path    = os.path.join(plot_dir, "per_horizon_metrics.png")


    # -----------------------------
    # Load checkpoint
    # -----------------------------
    checkpoint = torch.load(model_save_path, map_location='cpu')
    train_losses = checkpoint['train_losses']
    val_losses   = checkpoint['val_losses']
    best_epoch   = checkpoint['epoch']
    print(f"Checkpoint loaded (best epoch: {best_epoch}, val_loss: {checkpoint['val_loss']:.4f})")

    # -----------------------------
    # Load dataset and rebuild data loaders
    # -----------------------------
    dataset = torch.load(dataset_path, weights_only=False)
    encoder_data = dataset['encoder']
    decoder_data = dataset['decoder']
    target_data  = dataset['target']
    full_dataset = TensorDataset(encoder_data, decoder_data, target_data)

    val_ratio, test_ratio = 0.1, 0.1
    n_total    = len(full_dataset)
    test_size  = int(n_total * test_ratio)
    val_size   = int(n_total * val_ratio)
    train_size = n_total - val_size - test_size

    test_dataset = Subset(full_dataset, range(train_size + val_size, n_total))
    config       = Config()
    test_loader  = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # -----------------------------
    # Rebuild and load model
    # -----------------------------
    model = LSTMForecast(config).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # -----------------------------
    # Collect test predictions
    # Shape: (n_samples, 168)
    # -----------------------------
    all_preds_h, all_targets_h = [], []
    with torch.no_grad():
        for enc, dec, tgt in test_loader:
            enc, dec = enc.to(config.device), dec.to(config.device)
            mu, log_var = model(enc, dec)
            all_preds_h.append(mu.cpu().numpy())
            all_targets_h.append(tgt.numpy())

    preds_h   = np.concatenate(all_preds_h,   axis=0)
    targets_h = np.concatenate(all_targets_h, axis=0)

    # -----------------------------
    # Rescale predictions & targets
    # -----------------------------
    if "demand_mean" in dataset and "demand_std" in dataset:
        demand_mean = dataset["demand_mean"]
        demand_std  = dataset["demand_std"]
    else:
        # Dataset was created before demand_mean/demand_std were saved.
        # Estimate from the full target tensor as a fallback.
        # Re-run DatasetCreation.py to fix this permanently.
        print("WARNING: dataset.pt has no 'demand_mean'/'demand_std' keys. "
              "Estimating from target data — re-run DatasetCreation.py to fix this.")
        all_targets = target_data.detach().cpu().numpy()
        demand_mean = float(all_targets.mean())
        demand_std  = float(all_targets.std())

    preds_h   = preds_h * demand_std + demand_mean
    targets_h = targets_h * demand_std + demand_mean

    n_test_samples       = preds_h.shape[0]
    test_start_global_idx = train_size + val_size

    print(f"Generating plots ({n_test_samples} test samples)...")

    # Helper: add day-boundary markers to a subplot
    day_hours  = [24, 48, 72, 96, 120, 144, 168]
    day_labels = ['1d', '2d', '3d', '4d', '5d', '6d', '7d']

    def add_day_markers(ax):
        for h, lbl in zip(day_hours, day_labels):
            ax.axvline(x=h, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
            ax.annotate(lbl, xy=(h, 1.02), xycoords=('data', 'axes fraction'),
                        ha='center', fontsize=8, color='gray')

    # Helper: derive a date label from a test-set sample index
    def date_label_for(idx):
        global_idx   = test_start_global_idx + idx
        window_start = DATASET_START + pd.Timedelta(hours=global_idx + ENCODER_HISTORY)
        window_end   = window_start + pd.Timedelta(hours=167)
        return (f"{window_start.strftime('%Y-%m-%d %H:%M')} "
                f"→ {window_end.strftime('%Y-%m-%d %H:%M')}"), window_start

    # ================================================================
    # Plot 1: Train vs Validation Loss
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs_range = range(1, len(train_losses) + 1)
    ax.plot(epochs_range, train_losses, label="Train Loss",      linewidth=1.5, marker='o', markersize=2)
    ax.plot(epochs_range, val_losses,   label="Validation Loss", linewidth=1.5, marker='o', markersize=2)
    ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best epoch ({best_epoch})')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Train vs Validation Loss", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    all_losses = train_losses + val_losses
    if len(all_losses) > 0 and max(all_losses) / max(min(all_losses), 1e-10) > 10:
        ax.set_yscale('log')
        ax.set_ylabel("MSE Loss (log scale)", fontsize=12)

    plt.tight_layout()
    plt.savefig(train_val_plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {train_val_plot_path}")

    # ================================================================
    # Plot 2: Example Forecast Windows (3 windows, 1 row)
    # ================================================================
    sample_indices   = np.linspace(0, n_test_samples - 1, 3, dtype=int)
    hours            = np.arange(1, 169)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    for ax, idx in zip(axes, sample_indices):
        label, _ = date_label_for(idx)
        window_mae = np.mean(np.abs(targets_h[idx] - preds_h[idx]))
        ax.plot(hours, targets_h[idx], label='Actual',    linewidth=1.5, color='blue')
        ax.plot(hours, preds_h[idx],   label='Predicted', linewidth=1.5, color='red', alpha=0.8)
        ax.set_title(f"{label}\n(MAE: {window_mae:.4f})", fontsize=9)
        ax.set_xlabel("Forecast Hour", fontsize=9)
        ax.set_ylabel("abvaerk (MWh)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Example 168-Hour Forecast Windows (Test Set)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(test_plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {test_plot_path}")

    # ================================================================
    # Plot 3: Actual vs Predicted — 3 horizons side by side
    # ================================================================
    # Three scatter panels, one per horizon step:
    #   - h=0   → 1h ahead  (sharpest, most accurate)
    #   - h=23  → 24h ahead (1 day ahead)
    #   - h=167 → 168h ahead (7 days ahead, hardest)
    # Each panel plots one point per test window (no fan effect).
    # A small jitter separates overlapping points.
    # The identity line y=x and R² score show accuracy and bias at each horizon.
    rng = np.random.default_rng(42)

    def _scatter_panel(ax, horizon_idx, horizon_label):
        actual = targets_h[:, horizon_idx]
        pred   = preds_h[:, horizon_idx]
        jitter = rng.normal(0, 0.02, size=actual.shape)
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2     = 1 - ss_res / (ss_tot if ss_tot != 0 else 1e-10)
        ax.scatter(actual + jitter, pred, alpha=0.3, s=4, color='steelblue')
        lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
        ax.plot(lims, lims, 'k--', linewidth=1.0, label='y = x (perfect)')
        ax.set_xlabel("Actual abvaerk (MWh)", fontsize=11)
        ax.set_ylabel("Predicted abvaerk (MWh)", fontsize=11)
        ax.set_title(f"Actual vs Predicted — {horizon_label}\nR² = {r2:.4f}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig, axes_sc = plt.subplots(1, 3, figsize=(22, 7))
    _scatter_panel(axes_sc[0], horizon_idx=0,   horizon_label="1h Ahead (horizon 0)")
    _scatter_panel(axes_sc[1], horizon_idx=23,  horizon_label="24h Ahead (horizon 23)")
    _scatter_panel(axes_sc[2], horizon_idx=167, horizon_label="168h Ahead (horizon 167)")

    fig.suptitle("Actual vs Predicted at Three Forecast Horizons (Test Set)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(scatter_plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {scatter_plot_path}")

    # ================================================================
    # Plot 3: Residual Diagnostics  (3 panels)
    # ================================================================
    # Panel A — Residuals over time (MAE per window in chronological order).
    #   Reveals non-stationarity: error spikes at winter peaks, season
    #   transitions, or holidays that the model failed to generalise.
    #
    # Panel B — Error by day-of-week (box plot).
    #   Energy demand has strong weekly periodicity. If the model performs
    #   worse for certain starting days (e.g. Mondays after weekends), this
    #   exposes the pattern directly and maps to operational risk.
    #
    # Panel C — Quantile error heatmap (horizon × percentile).
    #   Shows the full distribution of absolute errors at every forecast
    #   step, not just the mean. Reveals whether tail errors grow faster
    #   than the median at longer horizons — critical for planning.

    # A: MAE per test window over time
    mae_per_window = np.mean(np.abs(targets_h - preds_h), axis=1)   # (n_samples,)
    window_dates   = [DATASET_START + pd.Timedelta(hours=(test_start_global_idx + i + ENCODER_HISTORY))
                      for i in range(n_test_samples)]

    # B: quantile error heatmap
    abs_errors  = np.abs(targets_h - preds_h)          # (n_samples, 168)
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    heatmap     = np.percentile(abs_errors, percentiles, axis=0)   # (7, 168)

    # C: predicted vs actual variance ratio per horizon.
    # A ratio near 1.0 means the model's spread matches reality.
    # Ratio < 1 means the model is under-dispersed (collapsing toward the
    # mean at that horizon — safe but uninformative predictions).
    # Ratio > 1 means the model is over-dispersed (exaggerating variation).
    pred_var   = np.var(preds_h,   axis=0)   # (168,)
    actual_var = np.var(targets_h, axis=0)   # (168,)
    var_ratio  = pred_var / np.where(actual_var == 0, 1e-10, actual_var)

    # Layout: 2 rows — top row spans full width (panel A),
    # bottom row has heatmap and variance ratio side by side
    fig = plt.figure(figsize=(18, 14))
    gs_r = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
                             height_ratios=[1, 1.2])

    # Panel A — residuals over time (spans full top row)
    ax_a = fig.add_subplot(gs_r[0, :])
    ax_a.plot(window_dates, mae_per_window, linewidth=0.8, color='steelblue', alpha=0.7)
    rolling = pd.Series(mae_per_window).rolling(window=168, min_periods=1).mean().values
    ax_a.plot(window_dates, rolling, linewidth=1.5, color='red', label='168-sample rolling mean')
    ax_a.set_xlabel("Forecast Start Date", fontsize=11)
    ax_a.set_ylabel("MAE (MWh)", fontsize=11)
    ax_a.set_title("Forecast Error Over Time — MAE per Window (Test Set)", fontsize=12)
    ax_a.legend(fontsize=9)
    ax_a.grid(True, alpha=0.3)

    # Panel B — quantile heatmap (bottom-left)
    ax_b = fig.add_subplot(gs_r[1, 0])
    im = ax_b.imshow(heatmap, aspect='auto', origin='lower',
                     extent=[1, 168, -0.5, len(percentiles) - 0.5],
                     cmap='YlOrRd')
    ax_b.set_yticks(range(len(percentiles)))
    ax_b.set_yticklabels([f"p{p}" for p in percentiles], fontsize=9)
    ax_b.set_xlabel("Forecast Horizon (hours)", fontsize=11)
    ax_b.set_ylabel("Error Percentile", fontsize=11)
    ax_b.set_title("Absolute Error Distribution by Horizon\n(quantile heatmap)", fontsize=12)
    for h in day_hours:
        ax_b.axvline(x=h, color='white', linestyle='--', linewidth=0.6, alpha=0.6)
    plt.colorbar(im, ax=ax_b, label='Absolute Error (MWh)')

    # Panel C — predicted vs actual variance ratio (bottom-right)
    ax_c = fig.add_subplot(gs_r[1, 1])
    ax_c.plot(range(1, 169), var_ratio, color='darkorange', linewidth=1.4)
    ax_c.axhline(1.0, color='black', linestyle='--', linewidth=0.9,
                 label='ratio = 1 (perfect dispersion)')
    ax_c.fill_between(range(1, 169), var_ratio, 1.0,
                      where=(var_ratio < 1.0), alpha=0.2, color='steelblue',
                      label='under-dispersed (mean regression)')
    ax_c.fill_between(range(1, 169), var_ratio, 1.0,
                      where=(var_ratio > 1.0), alpha=0.2, color='red',
                      label='over-dispersed')
    for h in day_hours:
        ax_c.axvline(x=h, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax_c.set_xlabel("Forecast Horizon (hours)", fontsize=11)
    ax_c.set_ylabel("Var(predicted) / Var(actual)", fontsize=11)
    ax_c.set_title("Predicted vs Actual Variance Ratio\nper Horizon", fontsize=12)
    ax_c.legend(fontsize=9)
    ax_c.grid(True, alpha=0.3)

    fig.suptitle("Residual Diagnostics (Test Set)", fontsize=14, fontweight='bold')
    plt.savefig(residuals_plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {residuals_plot_path}")

    # ================================================================
    # Plot 4: Per-Horizon Metrics with Persistence Baseline
    # ================================================================
    # MSE, RMSE, MAE, MAPE — each metric shows the LSTM line plus a
    # naive persistence baseline (predict last known encoder value).
    # The persistence baseline is the standard benchmark: if the LSTM
    # doesn't beat it, it's not adding value over a trivial model.
    mse_per_horizon  = np.mean((preds_h - targets_h) ** 2, axis=0)
    rmse_per_horizon = np.sqrt(mse_per_horizon)
    mae_per_horizon  = np.mean(np.abs(preds_h - targets_h), axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        mape_per_sample  = np.abs((targets_h - preds_h) / targets_h) * 100
        mape_per_sample  = np.where(np.isfinite(mape_per_sample), mape_per_sample, np.nan)
    mape_per_horizon = np.nanmean(mape_per_sample, axis=0)

    # R² per horizon: how much variance the model explains at each step
    ss_res_h = np.sum((targets_h - preds_h) ** 2, axis=0)
    ss_tot_h = np.sum((targets_h - targets_h.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2_per_horizon = 1 - ss_res_h / np.where(ss_tot_h == 0, 1e-10, ss_tot_h)

    # Build persistence baseline: last value of encoder = encoder_data[:, -1, 0]
    # encoder feature index 0 is 'abvaerk' (see DatasetCreation.py)
    test_encoder = encoder_data[train_size + val_size:]        # (n_test, 168, 8)
    last_known   = test_encoder[:, -1, 0].numpy()             # (n_test,)
    # Persistence prediction: repeat last known value for all 168 horizons
    persist_pred    = np.tile(last_known[:, None], (1, 168))  # (n_test, 168)
    persist_targets = targets_h

    persist_mse  = np.mean((persist_pred - persist_targets) ** 2, axis=0)
    persist_rmse = np.sqrt(persist_mse)
    persist_mae  = np.mean(np.abs(persist_pred - persist_targets), axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        persist_mape_s = np.abs((persist_targets - persist_pred) / persist_targets) * 100
        persist_mape_s = np.where(np.isfinite(persist_mape_s), persist_mape_s, np.nan)
    persist_mape = np.nanmean(persist_mape_s, axis=0)

    persist_ss_res = np.sum((persist_targets - persist_pred) ** 2, axis=0)
    persist_r2     = 1 - persist_ss_res / np.where(ss_tot_h == 0, 1e-10, ss_tot_h)

    fig, axes_h = plt.subplots(5, 1, figsize=(14, 20), sharex=True)
    hours_r = range(1, 169)

    metrics = [
        (axes_h[0], mse_per_horizon,  persist_mse,   'MSE',     'blue'),
        (axes_h[1], rmse_per_horizon, persist_rmse,  'RMSE',    'purple'),
        (axes_h[2], mae_per_horizon,  persist_mae,   'MAE',     'red'),
        (axes_h[3], mape_per_horizon, persist_mape,  'MAPE (%)', 'green'),
        (axes_h[4], r2_per_horizon,   persist_r2,    'R²',      'darkorange'),
    ]

    for ax, lstm_vals, pers_vals, ylabel, color in metrics:
        ax.plot(hours_r, lstm_vals, color=color,  linewidth=1.2, label='LSTM')
        ax.plot(hours_r, pers_vals, color='gray', linewidth=1.0, linestyle='--',
                label='Persistence baseline')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        add_day_markers(ax)

    # R² panel: add reference lines at 0.0 and 1.0.
    # R² = 0 means the model does no better than always predicting the mean.
    # R² = 1 means perfect prediction.
    # Beating the gray persistence line is the key comparison.
    axes_h[4].axhline(0.0, color='black', linestyle=':', linewidth=1.0, alpha=0.6,
                      label='R² = 0 (no better than mean)')
    axes_h[4].legend(fontsize=9)

    axes_h[0].set_title("Per-Horizon Forecast Error vs Persistence Baseline (Test Set)", fontsize=14)
    axes_h[4].set_xlabel("Forecast Horizon (hours)", fontsize=12)

    plt.tight_layout()
    plt.savefig(horizon_plot_path, dpi=150)
    plt.close()
    print(f"  Saved: {horizon_plot_path}")

    print("All plots saved successfully.")

    # ================================================================
    # Generate README
    # ================================================================
    generate_evaluation_readme(plot_dir, best_epoch, checkpoint['val_loss'], n_test_samples,
                               train_size, val_size, test_size, n_total,
                               model_filename=os.path.basename(run_dir) + "/model.pth")
    print(f"  Saved: {os.path.join(plot_dir, 'README_Evaluation.md')}")



# Allow standalone execution: python3 Plotting.py --local
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true',
                        help='Use local relative paths instead of server paths')
    args = parser.parse_args()
    main(local=args.local)

