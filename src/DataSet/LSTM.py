import json
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Path to dataset relative to this script
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / 'dataset.json'


class ForecastDataset(Dataset):
    """PyTorch Dataset for the forecast data."""

    def __init__(self, samples, targets):
        self.samples = torch.FloatTensor(samples)
        self.targets = torch.FloatTensor(targets).unsqueeze(1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


class LSTMModel(nn.Module):
    """Simple LSTM model for energy demand prediction."""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]

        # Pass through fully connected layers
        output = self.fc(last_hidden)
        return output


def load_dataset(filepath=DATASET_PATH):
    """Load the dataset from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    samples = np.array(data['samples'], dtype=np.float32)
    targets = np.array(data['targets'], dtype=np.float32)

    # Check for NaN and inf values
    print(f"\nData quality check:")
    print(f"  Samples NaN count: {np.isnan(samples).sum()}")
    print(f"  Samples Inf count: {np.isinf(samples).sum()}")
    print(f"  Targets NaN count: {np.isnan(targets).sum()}")
    print(f"  Targets Inf count: {np.isinf(targets).sum()}")
    print(f"  Samples min: {np.nanmin(samples)}, max: {np.nanmax(samples)}")
    print(f"  Targets min: {np.nanmin(targets)}, max: {np.nanmax(targets)}")

    # Remove samples with NaN or inf values
    valid_mask = ~np.isnan(targets) & ~np.isinf(targets)
    for i in range(samples.shape[2]):  # Check each feature
        feature_valid = ~np.any(np.isnan(samples[:, :, i]) | np.isinf(samples[:, :, i]), axis=1)
        valid_mask &= feature_valid

    samples = samples[valid_mask]
    targets = targets[valid_mask]

    print(f"\nLoaded dataset:")
    print(f"  Samples shape: {samples.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Features: {data['feature_names']}")
    print(f"  Removed {(~valid_mask).sum()} samples with NaN/Inf values")

    return samples, targets, data


def normalize_data(X_train, X_test, y_train, y_test):
    """Normalize features and targets using StandardScaler."""
    # Reshape for scaling: (n_samples * seq_len, n_features)
    n_train, seq_len, n_features = X_train.shape
    n_test = X_test.shape[0]

    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)

    # Scale features
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train_flat).reshape(n_train, seq_len, n_features)
    X_test_scaled = feature_scaler.transform(X_test_flat).reshape(n_test, seq_len, n_features)

    # Scale targets
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, feature_scaler, target_scaler


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """Train the LSTM model."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    return train_losses, val_losses


def evaluate_model(model, test_loader, target_scaler, device='cpu'):
    """Evaluate the model on test set."""
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(y_batch.numpy().flatten())

    # Inverse transform to get original scale
    predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = target_scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    print(f"\nTest Set Metrics:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    return predictions, actuals, {'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape}


def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    samples, targets, metadata = load_dataset()

    # Split data: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        samples, targets, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # Normalize data
    X_train, X_val, y_train, y_val, feature_scaler, target_scaler = normalize_data(
        X_train, X_val, y_train, y_val
    )
    X_test_norm = feature_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    y_test_norm = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    # Create data loaders
    batch_size = 32
    train_dataset = ForecastDataset(X_train, y_train)
    val_dataset = ForecastDataset(X_val, y_val)
    test_dataset = ForecastDataset(X_test_norm, y_test_norm)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_size = metadata['n_features']
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2)

    print(f"\nModel architecture:")
    print(model)

    # Train model
    print(f"\nTraining...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader,
        epochs=50, lr=0.001, device=device
    )

    # Evaluate on test set
    predictions, actuals, metrics = evaluate_model(model, test_loader, target_scaler, device)

    # Save model
    model_path = SCRIPT_DIR / 'lstm_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'metrics': metrics,
        'input_size': input_size
    }, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == '__main__':
    main()




