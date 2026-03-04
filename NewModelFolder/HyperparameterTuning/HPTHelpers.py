import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import os
from tqdm import tqdm
import optuna
from optuna.trial import Trial
import json
from datetime import datetime
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LSTMModel import Config, LSTMForecast
from HyperparameterTuning import HPTOutput as output


def train_model(config, train_loader, val_loader, train_size, val_size, device, 
                trial=None, max_epochs=None, verbose=False):
    """
    Train the LSTM model with given hyperparameters.
    
    Args:
        config: Config object with hyperparameters
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        train_size: Number of training samples
        val_size: Number of validation samples
        device: Device to train on
        trial: Optuna trial object (optional, for pruning)
        max_epochs: Maximum epochs to train (overrides config.epochs if provided)
        verbose: If True, print training progress
    
    Returns:
        best_val_loss: Best validation loss achieved
        model: Trained model
    """
    if verbose and trial is not None:
        output.print_trial_info(trial, config)
    
    model = LSTMForecast(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Early stopping
    patience = 10  # Reduced for hyperparameter tuning
    best_val_loss = np.inf
    epochs_no_improve = 0
    best_model_state = None
    
    epochs = max_epochs if max_epochs is not None else config.epochs
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        epoch_loss = 0
        
        # Add progress bar for first trial or if verbose
        show_progress = verbose or (trial and trial.number == 0)
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", 
                         leave=False) if show_progress else train_loader
        
        for enc, dec, tgt in train_iter:
            enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(enc, dec)
            loss = criterion(output, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * enc.size(0)
        
        train_loss = epoch_loss / train_size
        
        # Validation
        model.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for enc, dec, tgt in val_loader:
                enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
                output = model(enc, dec)
                val_loss_epoch += criterion(output, tgt).item() * enc.size(0)
        val_loss = val_loss_epoch / val_size
        
        scheduler.step(val_loss)
        
        # Print progress for first few epochs or if verbose
        if verbose and epoch <= 3:
            output.print_epoch_info(epoch, train_loss, val_loss)
        
        # Report to Optuna for pruning (if trial is provided)
        if trial is not None:
            trial.report(val_loss, epoch)
            # Check if trial should be pruned
            if trial.should_prune():
                if verbose:
                    output.print_trial_pruned(epoch)
                raise optuna.TrialPruned()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    output.print_early_stopping(epoch)
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return best_val_loss, model


def objective(trial: Trial, train_dataset, val_dataset, device, 
              local=False, verbose=False):
    """
    Objective function for Optuna to optimize.
    
    Args:
        trial: Optuna trial object
        train_dataset: Training dataset
        val_dataset: Validation dataset
        device: Device to train on
        local: Whether running in local mode
        verbose: Whether to print detailed progress
    
    Returns:
        best_val_loss: Best validation loss (to minimize)
    """
    
    # Create config with suggested hyperparameters
    config = Config()
    
    # Suggest hyperparameters
    config.hidden_size = trial.suggest_int('hidden_size', 64, 256, step=32)
    config.num_layers = trial.suggest_int('num_layers', 1, 4)
    config.dropout = trial.suggest_float('dropout', 0.1, 0.5)
    config.batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    config.learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    # Optional: tune sequence lengths (commented out by default)
    # config.encoder_history = trial.suggest_int('encoder_history', 72, 336, step=24)
    # config.forecast_length = trial.suggest_int('forecast_length', 72, 336, step=24)
    
    config.device = device
    
    # Create data loaders
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                            shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                          shuffle=False)
    
    try:
        # Train model with early stopping
        best_val_loss, _ = train_model(
            config, train_loader, val_loader, train_size, val_size, 
            device, trial=trial, max_epochs=50, verbose=verbose
        )
        
        return best_val_loss
    
    except Exception as e:
        output.print_trial_failed(e)
        return float('inf')


def load_dataset(local=False, filePaths=None):
    """
    Load and split dataset into train/val/test sets.
    
    Args:
        dataset_path: Path to dataset.pt file
        local: Whether running in local mode
    
    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset (not used in tuning)
    """
    # Setup paths
    dataset_path = filePaths[0]

    # Load dataset
    output.print_loading_dataset(dataset_path)
    
    if not os.path.exists(dataset_path):
        print(f"\nError: Dataset file not found at: {dataset_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
        print("\nPlease either:")
        print("  1. Provide the correct path with --dataset <path>")
        print("  2. Ensure dataset.pt is in the correct location")
        sys.exit(1)
    
    dataset = torch.load(dataset_path, weights_only=True)
    encoder_data = dataset['encoder']
    decoder_data = dataset['decoder']
    target_data = dataset['target']
    full_dataset = TensorDataset(encoder_data, decoder_data, target_data)
    
    # Train/Val/Test split
    val_ratio = 0.1
    test_ratio = 0.1
    n_total = len(full_dataset)
    test_size = int(n_total * test_ratio)
    val_size = int(n_total * val_ratio)
    train_size = n_total - val_size - test_size
    
    train_dataset = Subset(full_dataset, range(0, train_size))
    val_dataset = Subset(full_dataset, range(train_size, train_size + val_size))
    test_dataset = Subset(full_dataset, range(train_size + val_size, n_total))
    
    output.print_dataset_info(train_size, val_size, test_size)
    
    return train_dataset, val_dataset, test_dataset


def get_results_dir(local=False):
    """Get the appropriate results directory based on mode."""
    if local:
        return os.path.dirname(os.path.abspath(__file__))
    else:
        return "/ceph/project/SW6-Group18-Abvaerk/ServerReady"


def run_hyperparameter_search(n_trials=50, 
                              local=False, 
                              verbose=False,
                              filePaths=None):
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        n_trials: Number of trials to run
        local: Whether running in local mode
        dataset_path: Path to dataset (optional, auto-detected if None)
        verbose: Whether to print detailed progress
    
    Returns:
        study: Optuna study object with results
    """
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output.print_device_info(device)
    
    # Load dataset
    train_dataset, val_dataset, _ = load_dataset(local, filePaths)
    
    # Create Optuna study
    study_name = f"lstm_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    output.print_tuning_start(n_trials)
    
    if verbose:
        output.print_verbose_mode()
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, train_dataset, val_dataset, device, 
                              local, verbose),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Print results
    output.print_tuning_complete()
    output.print_best_trial(study.best_trial)
    
    # Save best parameters as JSON
    # Determine where to save/load the hyperparameter JSON

    best_params_file = filePaths[1]
    
    os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
    with open(best_params_file, 'w') as f:
        json.dump(study.best_trial.params, f, indent=2)
    
    return study