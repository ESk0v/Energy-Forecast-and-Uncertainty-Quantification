# HyperParameterTuning/LearningRateTuning.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random

def lr_range_test(
        model_class, 
        model_kwargs, 
        train_loader, 
        criterion, 
        device,
        lr_start=1e-5, 
        lr_end=1, 
        num_iters=100, 
        seed=42, 
        subset_size=None):
    
    _SetSeed(seed)
    model = model_class(**model_kwargs).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_start)

    # LR multiplier per iteration
    lr_mult = (lr_end / lr_start) ** (1 / num_iters)

    lrs = []
    losses = []
    model.train()
    iter_count = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if iter_count >= num_iters:
            break
        if subset_size and batch_idx >= subset_size:
            break

        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())

        optimizer.param_groups[0]['lr'] *= lr_mult
        iter_count += 1

    best_lr = _find_best_lr(lrs, losses)

    print("\n--- LRRT Results ---")
    print("Learning Rates tested:", lrs)
    print("Corresponding losses:", losses)
    print(f"Suggested best learning rate: {best_lr:.6f}")
    return lrs, losses, best_lr

def _find_best_lr(lrs, losses, smooth_window=5, skip_start=5, skip_end=5):
    """
    Find the best LR by locating the point of steepest loss descent
    on a smoothed curve, ignoring the noisy start and diverging tail.

    Strategy:
      1. Smooth the loss curve with a moving average to reduce noise.
      2. Clip off the first `skip_start` and last `skip_end` points
         (early chaos and late divergence are not useful).
      3. Compute the gradient of the smoothed curve.
      4. Pick the LR one step before the steepest negative gradient
         (i.e. where the loss is dropping fastest — the sweet spot).
    """
    losses_arr = np.array(losses)
    lrs_arr    = np.array(lrs)

    # Step 1: smooth
    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.convolve(losses_arr, kernel, mode='same')

    # Step 2: clip noisy edges
    start = skip_start
    end   = len(smoothed) - skip_end
    smoothed_clipped = smoothed[start:end]
    lrs_clipped      = lrs_arr[start:end]

    if len(smoothed_clipped) == 0:
        # Fallback: just return the raw minimum if clipping ate everything
        return lrs_arr[np.argmin(losses_arr)]

    # Step 3: gradient
    gradients = np.gradient(smoothed_clipped)

    # Step 4: steepest descent point, then step one back (safer side of the drop)
    steepest_idx = np.argmin(gradients)
    best_idx     = max(0, steepest_idx - 1)

    return float(lrs_clipped[best_idx])

def plot_lr_results(lrs, losses, save_path=None, title="Learning Rate Range Test"):
    best_lr = _find_best_lr(lrs, losses)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lrs, losses, marker='o', markersize=3, linewidth=1.2, label="Loss")

    # Mark best LR
    ax.axvline(x=best_lr, color='red', linestyle='--', linewidth=1.5,
               label=f"Best LR: {best_lr:.6f}")

    ax.set_xscale('log')
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"LRRT plot saved to {save_path}")

def _SetSeed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False