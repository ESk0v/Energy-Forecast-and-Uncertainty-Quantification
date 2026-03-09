import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import argparse
import os
from tqdm import tqdm
import optuna
from optuna.trial import Trial
import json
from datetime import datetime

from LSTMModel import Config, LSTMForecast


def train_model(config, train_loader, val_loader, train_size, val_size, device, trial=None, max_epochs=None, verbose=False):
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
        print(f"\n  Trial {trial.number}: hidden_size={config.hidden_size}, num_layers={config.num_layers}, "
              f"dropout={config.dropout:.2f}, batch_size={config.batch_size}, lr={config.learning_rate:.6f}")
    
    model = LSTMForecast(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Early stopping
    patience = 5  # Reduced for hyperparameter tuning
    best_val_loss = np.inf
    epochs_no_improve = 0
    best_model_state = None
    
    epochs = max_epochs if max_epochs is not None else config.epochs
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        epoch_loss = 0
        
        # Add progress bar for first trial or if verbose
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False) if (verbose or (trial and trial.number == 0)) else train_loader
        
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
            print(f"    Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Report to Optuna for pruning (if trial is provided)
        if trial is not None:
            trial.report(val_loss, epoch)
            # Check if trial should be pruned
            if trial.should_prune():
                if verbose:
                    print(f"    Trial pruned at epoch {epoch}")
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
                    print(f"    Early stopped at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return best_val_loss, model


def objective(trial: Trial, train_dataset, val_dataset, device, local=False, verbose=False):
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
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    try:
        # Train model with early stopping
        best_val_loss, _ = train_model(
            config, train_loader, val_loader, train_size, val_size, 
            device, trial=trial, max_epochs=20, verbose=verbose  # Max epochs for tuning
        )
        
        return best_val_loss
    
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('inf')


def run_hyperparameter_search(n_trials=50, local=False, dataset_path=None, verbose=False):
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
    
    # Setup paths
    if local:
        _dir = os.path.dirname(os.path.abspath(__file__))
        if dataset_path is None:
            dataset_path = os.path.join(_dir, "dataset.pt")
        results_dir = _dir
        print("Running in LOCAL mode (relative paths)")
    else:
        if dataset_path is None:
            dataset_path = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/dataset.pt"
        results_dir = "/ceph/project/SW6-Group18-Abvaerk/ServerReady"
        print("Running in SERVER mode (absolute paths)")
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path, weights_only=True)
    encoder_data = dataset['encoder']
    decoder_data = dataset['decoder']
    target_data = dataset['target']
    full_dataset = TensorDataset(encoder_data, decoder_data, target_data)
    
    # Train/Val split (we'll use the same split as training for consistency)
    val_ratio = 0.1
    test_ratio = 0.1
    n_total = len(full_dataset)
    test_size = int(n_total * test_ratio)
    val_size = int(n_total * val_ratio)
    train_size = n_total - val_size - test_size
    
    train_dataset = Subset(full_dataset, range(0, train_size))
    val_dataset = Subset(full_dataset, range(train_size, train_size + val_size))
    
    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Create Optuna study
    study_name = f"lstm_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    print(f"\nStarting hyperparameter search with {n_trials} trials...")
    print("=" * 70)
    
    if verbose:
        print("\n(Verbose mode: showing detailed training progress)\n")
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, train_dataset, val_dataset, device, local, verbose),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 70)
    print(f"\nBest trial:")
    trial = study.best_trial
    print(f"  Validation Loss: {trial.value:.6f}")
    print(f"\n  Best Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save results
    results_file = os.path.join(results_dir, f"{study_name}_results.csv")
    df = study.trials_dataframe()
    df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")
    
    # Save best parameters as JSON
    best_params_file = os.path.join(results_dir, f"{study_name}_best_params.json")
    with open(best_params_file, 'w') as f:
        json.dump(trial.params, f, indent=2)
    print(f"Best parameters saved to: {best_params_file}")
    
    # Generate visualizations if possible
    try:
        import optuna.visualization as vis
        
        viz_dir = os.path.join(results_dir, "optuna_visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Optimization history
        fig1 = vis.plot_optimization_history(study)
        fig1.write_html(os.path.join(viz_dir, 'optimization_history.html'))
        
        # Parameter importance
        fig2 = vis.plot_param_importances(study)
        fig2.write_html(os.path.join(viz_dir, 'param_importances.html'))
        
        # Parallel coordinate plot
        fig3 = vis.plot_parallel_coordinate(study)
        fig3.write_html(os.path.join(viz_dir, 'parallel_coordinate.html'))
        
        # Slice plot
        fig4 = vis.plot_slice(study)
        fig4.write_html(os.path.join(viz_dir, 'slice_plot.html'))
        
        print(f"\nVisualizations saved to: {viz_dir}")
        
    except Exception as e:
        print(f"\nNote: Could not generate visualizations: {e}")
        print("Install plotly for visualizations: pip install plotly")
    
    return study


def train_with_best_params(study, local=False, dataset_path=None):
    """
    Train a final model using the best hyperparameters found.
    
    Args:
        study: Optuna study object with completed trials
        local: Whether running in local mode
        dataset_path: Path to dataset (optional)
    """
    
    # Setup paths
    if local:
        _dir = os.path.dirname(os.path.abspath(__file__))
        if dataset_path is None:
            dataset_path = os.path.join(_dir, "dataset.pt")
        model_save_path = os.path.join(_dir, "best_tuned_lstm_model.pth")
    else:
        if dataset_path is None:
            dataset_path = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/dataset.pt"
        model_save_path = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/best_tuned_lstm_model.pth"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset
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
    
    # Create config with best parameters
    config = Config()
    best_params = study.best_trial.params
    
    config.hidden_size = best_params['hidden_size']
    config.num_layers = best_params['num_layers']
    config.dropout = best_params['dropout']
    config.batch_size = best_params['batch_size']
    config.learning_rate = best_params['learning_rate']
    config.device = device
    config.epochs = 100  # Train longer for final model
    
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("=" * 70)
    print("\nHyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Train final model
    best_val_loss, model = train_model(
        config, train_loader, val_loader, train_size, val_size, 
        device, trial=None, max_epochs=config.epochs
    )
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(config),
        'best_params': best_params,
        'val_loss': best_val_loss,
    }, model_save_path)
    
    print(f"\nFinal model saved to: {model_save_path}")
    print(f"Final validation loss: {best_val_loss:.6f}")
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description='LSTM Hyperparameter Tuning with Optuna')
    parser.add_argument('--local', action='store_true', 
                       help='Use local relative paths instead of server paths')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of trials for hyperparameter search (default: 50)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to dataset.pt file (optional)')
    parser.add_argument('--train_final', action='store_true',
                       help='Train final model with best parameters after tuning')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test run with only 10 trials')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed training progress (helpful for debugging)')
    
    args = parser.parse_args()
    
    # Adjust trials for quick test
    n_trials = 10 if args.quick else args.n_trials
    
    # Run hyperparameter search
    study = run_hyperparameter_search(
        n_trials=n_trials,
        local=args.local,
        dataset_path=args.dataset,
        verbose=args.verbose
    )
    
    # Optionally train final model
    if args.train_final:
        print("\n" + "=" * 70)
        train_with_best_params(study, local=args.local, dataset_path=args.dataset)
    else:
        print("\n" + "=" * 70)
        print("To train a final model with the best parameters, run:")
        print(f"  python HyperparameterTuning.py {'--local' if args.local else ''} --train_final")
        print("=" * 70)


if __name__ == "__main__":
    main()