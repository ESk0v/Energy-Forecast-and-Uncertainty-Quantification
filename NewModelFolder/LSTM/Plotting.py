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
        # Propagated from Main.py — use as-is
        dataset_path = filePaths[0]
        model_dir    = filePaths[1]
        plot_dir     = filePaths[2]
    else:
        # Standalone fallback (python3 Plotting.py --local)
        base_dir     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        dataset_path = os.path.join(base_dir, "Files", "dataset.pt")
        model_dir    = os.path.join(base_dir, "Models", "SingleLSTM")
        plot_dir     = os.path.join(base_dir, "Plots")

    print(f"Plotting — dataset : {dataset_path}")
    print(f"Plotting — model   : {model_dir}")
    print(f"Plotting — output  : {plot_dir}")

    os.makedirs(plot_dir, exist_ok=True)
    train_val_plot_path  = os.path.join(plot_dir, "train_val_loss.png")
    test_plot_path       = os.path.join(plot_dir, "test_predictions.png")
    scatter_plot_path    = os.path.join(plot_dir, "actual_vs_predicted.png")
    residuals_plot_path  = os.path.join(plot_dir, "residuals.png")
    horizon_plot_path    = os.path.join(plot_dir, "per_horizon_metrics.png")

    # Find the latest versioned model: model_v1.pth, model_v2.pth, ...
    existing = [f for f in os.listdir(model_dir) if f.startswith("model_v") and f.endswith(".pth")]
    existing_versions = []
    for f in existing:
        try:
            v = int(f.replace("model_v", "").replace(".pth", ""))
            existing_versions.append(v)
        except ValueError:
            pass
    if not existing_versions:
        raise FileNotFoundError(f"No versioned models found in {model_dir}")
    latest_version = max(existing_versions)
    model_save_path = os.path.join(model_dir, f"model_v{latest_version}.pth")
    print(f"Loading latest model: model_v{latest_version}.pth")

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
    _generate_readme(plot_dir, best_epoch, checkpoint['val_loss'], n_test_samples,
                     train_size, val_size, test_size, n_total,
                     model_filename=os.path.basename(model_save_path))
    print(f"  Saved: {os.path.join(plot_dir, 'README.md')}")


def _generate_readme(plot_dir, best_epoch, best_val_loss,
                     n_test_samples, train_size, val_size, test_size, n_total,
                     model_filename="unknown"):
    """
    Write a README.md to plot_dir explaining every plot, every panel,
    and how to interpret patterns in them.
    """
    train_pct = train_size / n_total * 100
    val_pct   = val_size   / n_total * 100
    test_pct  = test_size  / n_total * 100

    content = f"""\
# LSTM Forecast — Evaluation Plots

Auto-generated by `Plotting.py`.

## Model & Dataset Summary

| Property | Value |
|---|---|
| Model file | `{model_filename}` |
| Best checkpoint epoch | {best_epoch} |
| Best validation loss (MSE) | {best_val_loss:.6f} |
| Total samples | {n_total} |
| Train split | {train_size} ({train_pct:.1f}%) |
| Validation split | {val_size} ({val_pct:.1f}%) |
| Test split | {test_size} ({test_pct:.1f}%) |
| Forecast horizon | 168 hours (7 days) |
| Encoder history | 168 hours (7 days) |

---

## Plots Overview

| File | Contents |
|---|---|
| `train_val_loss.png` | Training and validation loss curves |
| `test_predictions.png` | Three representative 168-hour forecast windows |
| `actual_vs_predicted.png` | Actual vs predicted scatter at 1h, 24h, and 168h ahead |
| `residuals.png` | Residual diagnostics: error over time, quantile heatmap, variance ratio |
| `per_horizon_metrics.png` | Per-horizon MSE / RMSE / MAE / MAPE / R² vs persistence baseline |

---

## `train_val_loss.png` — Train vs Validation Loss

### What it shows
The MSE loss on the training set (blue) and validation set (orange) plotted against epoch
number. A vertical green dashed line marks the epoch at which the best validation loss was
achieved and the model checkpoint was saved.

The y-axis switches to log scale automatically if the loss range spans more than one order
of magnitude (common in early training when loss drops steeply).

### How to interpret it

| Pattern | Meaning |
|---|---|
| Both curves decrease together | Training is working correctly |
| Validation loss levels off while train loss keeps falling | Overfitting — the model is memorising training data |
| Validation loss is consistently *lower* than train loss | Unusual; can happen with dropout active only during training |
| Both curves are flat / not decreasing | Learning rate too low, or model not expressive enough |
| Loss spikes mid-training | Gradient explosion — consider lowering learning rate or clipping |
| Very few epochs on the x-axis | Model was stopped early (early stopping triggered) or epochs setting is low |

A good training run shows both curves decreasing and converging, with validation loss
slightly above train loss and the gap not growing over time.

---

## `test_predictions.png` — Example Forecast Windows

### What it shows
Three 168-hour forecast windows sampled evenly from the test set (early, middle, late).
Each panel shows the actual abvaerk (blue) and the model prediction (red) over the full
7-day horizon. The MAE and date range for that window are shown in the title.

### How to interpret

| Pattern | Meaning |
|---|---|
| Red line tracks blue closely | Model is working well for that period |
| Red line is flat / near-constant | Model is predicting the mean — under-trained or mean-regressing |
| Red line tracks the shape but is shifted up or down | Systematic bias for that period |
| Errors are larger in one window than another | Model performs differently across seasons or weather regimes |
| Red line overshoots peaks | Model is over-dispersed for that window |

The three windows are spaced across the full test set so you can compare performance
across different time periods (e.g. early autumn vs late winter).

---

## `actual_vs_predicted.png` — Actual vs Predicted at Three Horizons

### What it shows
Three scatter plots side by side, each showing actual abvaerk (x-axis) against the
model's predicted abvaerk (y-axis) for a specific forecast horizon step. Each point
represents one test window. The dashed diagonal is the identity line `y = x` (perfect
prediction). The R² score for that horizon is shown in the title.

| Panel | Horizon | Description |
|---|---|---|
| Left | h=0 (1h ahead) | Sharpest prediction — model has the most context |
| Middle | h=23 (24h ahead) | 1-day-ahead prediction |
| Right | h=167 (168h ahead) | 7-day-ahead prediction — hardest step |

A small random jitter is added to the x-axis to separate overlapping points and reveal
density structure that would otherwise be hidden.

### How to interpret

| Pattern | Meaning |
|---|---|
| Points clustered tightly on the diagonal | High accuracy at that horizon |
| Scatter grows from left to middle to right panel | Normal degradation — model gets less accurate further out |
| Points form a fan shape (wider at high actual values) | Heteroscedastic error — model struggles more at peak demand |
| Points systematically above the diagonal | Model over-predicts on average at that horizon |
| Points systematically below the diagonal | Model under-predicts on average at that horizon |
| Middle and right panels look similar to the left | Model is not degrading with horizon — strong weather forecast features or mean regression |
| Two distinct clusters of points | Possible bi-modal demand (e.g. heating season vs non-heating) |
| R² drops sharply from left to right | Expected — forecast accuracy degrades with horizon distance |
| R² stays similar across all three panels | Either genuinely good long-range skill, or mean regression (check variance ratio plot) |

---

## `residuals.png` — Residual Diagnostics

### Panel layout
```
┌──────────────────────────────────────┐
│  Panel A: MAE over time (full width) │
├──────────────────────┬───────────────┤
│  Panel B: Quantile   │  Panel C:     │
│  error heatmap       │  Variance     │
│                      │  ratio        │
└──────────────────────┴───────────────┘
```

### Panel A — Forecast Error Over Time
MAE for each test window plotted in chronological order (one point = one 168-hour forecast
starting at that date). A red rolling mean (168-sample window) is overlaid to show the
trend.

**How to interpret:**

| Pattern | Meaning |
|---|---|
| Gradual upward trend over time | Model degrades on later data — possible distribution shift |
| Sharp spike at a specific date | Model failed on an unusual event (cold snap, heat wave, holiday) |
| Lower error in summer, higher in winter | Model handles low-demand periods well but struggles with heating season |
| Error decreases over time | Test set may start mid-season transition and stabilise |
| Flat, consistent error throughout | Model generalises well across all periods |

The November transition spike (if present) is a classic heating season onset pattern for
district heating — sudden demand increases that the model has not seen enough of in training.

### Panel B — Absolute Error Distribution by Horizon (quantile heatmap)
A 2D heatmap where:
- **X-axis**: forecast horizon in hours (1 = 1h ahead, 168 = 7 days ahead)
- **Y-axis**: error percentile (p5 = best 5% of windows, p95 = worst 5%)
- **Colour**: absolute error in MWh (yellow = low, dark red = high)

Vertical white dashed lines mark day boundaries (1d, 2d, … 7d).

**How to interpret:**

| Pattern | Meaning |
|---|---|
| Colour gets darker left to right | Error grows with horizon — normal degradation |
| Colour is uniform across the full x-axis | Error does not grow with horizon — either genuinely good (weather forecasts help) or mean-regression (model predicts average regardless of horizon) |
| Only the top rows (p90, p95) darken at long horizons | Tail errors grow faster than median — worst-case scenarios are harder to predict far out |
| All rows darken uniformly | Error distribution shifts up uniformly with horizon |
| Dark colours near the top even at short horizons | Large worst-case errors even at 1–2h ahead — high-variance regime |

A flat heatmap that is uniformly coloured across the x-axis is ambiguous on its own.
Cross-reference with the variance ratio plot (Panel C) to distinguish a genuinely
flat-error model from one that is regressing to the mean.

### Panel C — Predicted vs Actual Variance Ratio per Horizon
The ratio `Var(predicted) / Var(actual)` computed at each of the 168 forecast steps.
The dashed black line at 1.0 is the ideal. Blue shading = under-dispersed region;
red shading = over-dispersed region.

**How to interpret:**

| Pattern | Meaning |
|---|---|
| Orange line stays near 1.0 across all horizons | Model maintains realistic spread at all forecast distances — genuine skill |
| Line drops below 1.0 after a few hours and keeps falling | Mean regression: model is predicting near-average values at longer horizons — predictions look plausible but miss peaks and troughs |
| Line starts above 1.0 (red zone) at horizon 1–5 | Decoder is over-dispersed initially — common in under-trained models |
| Line is flat below 1.0 for the entire range | Model systematically under-represents variation at all horizons |
| Line oscillates around 1.0 | Variance tracking is noisy but unbiased on average |

**Example — well-trained model:** Orange line hovers between 0.95–1.05 across all horizons.

**Example — mean-regressing model:** Line starts near 1.0 then steadily falls to ~0.7–0.8
by horizon 168. The model needs an attention mechanism or more training data.

**Example — under-trained model:** Line spikes above 1.0 at horizons 1–5, then drops
sharply below 1.0 and stays there. Both effects shrink after sufficient training.

---

## `per_horizon_metrics.png` — Per-Horizon Metrics vs Persistence Baseline

### What it shows
Five vertically stacked subplots sharing the same x-axis (forecast horizon 1–168h).
Each shows the LSTM metric (coloured line) and the **persistence baseline** (gray dashed
line). Vertical gray dashed lines mark day boundaries (1d–7d).

The persistence baseline naively repeats the last known abvaerk value for all 168 steps.
It is a strong baseline at short horizons but degrades quickly beyond 24h.

### Metrics explained

| Metric | Unit | Ideal direction | Formula |
|---|---|---|---|
| MSE | MWh² | Lower is better | mean((pred − actual)²) |
| RMSE | MWh | Lower is better | sqrt(MSE) — same unit as abvaerk |
| MAE | MWh | Lower is better | mean(abs(pred − actual)) |
| MAPE | % | Lower is better | mean(abs(pred − actual) / actual) × 100 |
| R² | dimensionless | Higher is better (max 1.0) | 1 - SS_res / SS_tot |

The dotted black line on the R² panel marks R² = 0 — below this the model is worse than
always predicting the mean.

### How to interpret each pattern

**MSE / RMSE / MAE:**

| Pattern | Meaning |
|---|---|
| LSTM line clearly below baseline at all horizons | Strong model — outperforms persistence at every step |
| LSTM below baseline at short horizons, above at long | Model has skill in the near term but loses it at multi-day distances |
| LSTM above baseline everywhere | Model is worse than naive persistence — undertrained or structural issue |
| Both lines flat (no growth with horizon) | Unusual; may indicate mean regression in both model and baseline |
| Both lines rise steeply after 24h | Normal degradation pattern for complex multi-day forecasts |

**MAPE:**
MAPE is sensitive to near-zero actual values. Spikes in the MAPE panel that don't
appear in MAE may indicate timesteps where actual abvaerk was very low (e.g. overnight
in summer).

**R²:**

| Pattern | Meaning |
|---|---|
| LSTM R² above baseline R² at all horizons | LSTM explains more variance than naive persistence everywhere ✓ |
| LSTM R² above 0 but below baseline | Model adds some signal but less than persistence — borderline useful |
| LSTM R² drops below 0 at long horizons | Model is worse than predicting the mean at those distances |
| Both LSTM and baseline R² near 0 at all horizons | High-variance demand — both models struggle to explain it |
| R² stays flat across all horizons | Model maintains skill across the full 7-day window |

---

## General notes

- All metrics are computed on the **test set only** ({test_size} samples, {test_pct:.1f}%
  of the data — chronologically the most recent). The model never saw this data during training.
- The persistence baseline uses the **last observed abvaerk value** from the encoder
  window repeated for all 168 forecast steps.
- A model trained for only **1–2 epochs** will typically show: high error, variance ratio
  well below 1.0 at long horizons, R² near or below the persistence baseline, and forecast
  windows that look flat or mean-like. These are training artefacts — always re-evaluate
  after full training.
"""
    readme_path = os.path.join(plot_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)


# Allow standalone execution: python3 Plotting.py --local
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true',
                        help='Use local relative paths instead of server paths')
    args = parser.parse_args()
    main(local=args.local)

