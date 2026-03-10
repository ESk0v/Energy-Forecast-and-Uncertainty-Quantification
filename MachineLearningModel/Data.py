import json
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class ForecastDataset(Dataset):
    """PyTorch Dataset wrapping feature sequences and scalar targets."""

    def __init__(self, samples, targets):
        self.samples = torch.FloatTensor(samples)
        self.targets = torch.FloatTensor(targets).unsqueeze(1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

def load_dataset(filepath, verbose=True):
    """
    Load, validate, and clean the dataset from a JSON file.

    Returns:
        samples  : np.ndarray of shape (n_samples, seq_len, n_features)
        targets  : np.ndarray of shape (n_samples,)
        metadata : dict with feature names, counts, etc.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    samples = np.array(data['samples'], dtype=np.float32)
    targets = np.array(data['targets'], dtype=np.float32)

    if verbose:
        # --- Data quality report ---
        print("\nData quality check:")
        print(f"  Samples NaN count : {np.isnan(samples).sum()}")
        print(f"  Samples Inf count : {np.isinf(samples).sum()}")
        print(f"  Targets NaN count : {np.isnan(targets).sum()}")
        print(f"  Targets Inf count : {np.isinf(targets).sum()}")
        print(f"  Samples range     : [{np.nanmin(samples):.4f}, {np.nanmax(samples):.4f}]")
        print(f"  Targets range     : [{np.nanmin(targets):.4f}, {np.nanmax(targets):.4f}]")

    # --- Remove invalid rows ---
    valid_mask = ~np.isnan(targets) & ~np.isinf(targets)
    for i in range(samples.shape[2]):
        feature_valid = ~np.any(np.isnan(samples[:, :, i]) | np.isinf(samples[:, :, i]), axis=1)
        valid_mask &= feature_valid

    samples = samples[valid_mask]
    targets = targets[valid_mask]

    if verbose:
        # --- Summary ---
        print(f"\nLoaded dataset:")
        print(f"  Samples shape : {samples.shape}")
        print(f"  Targets shape : {targets.shape}")
        print(f"  Features      : {data['feature_names']}")
        print(f"  Removed       : {(~valid_mask).sum()} invalid samples")

    return samples, targets, data


def normalize_data(X_train, X_val, y_train, y_val):
    """
    Fit scalers on training data and apply to both train and validation sets.

    Reshapes internally to handle the 3D sample array (n_samples, seq_len, n_features).

    Returns:
        X_train_scaled, X_val_scaled   : normalized feature arrays
        y_train_scaled, y_val_scaled   : normalized target arrays
        feature_scaler, target_scaler  : fitted StandardScaler instances
    """
    n_train, seq_len, n_features = X_train.shape
    n_val = X_val.shape[0]

    # Flatten time dimension for sklearn, then restore shape after scaling
    X_train_scaled = (
        StandardScaler()
        .fit(X_train.reshape(-1, n_features))
        .transform(X_train.reshape(-1, n_features))
        .reshape(n_train, seq_len, n_features)
    )

    feature_scaler = StandardScaler().fit(X_train.reshape(-1, n_features))
    X_train_scaled = feature_scaler.transform(X_train.reshape(-1, n_features)).reshape(n_train, seq_len, n_features)
    X_val_scaled   = feature_scaler.transform(X_val.reshape(-1, n_features)).reshape(n_val, seq_len, n_features)

    target_scaler  = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled   = target_scaler.transform(y_val.reshape(-1, 1)).flatten()

    return X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, feature_scaler, target_scaler