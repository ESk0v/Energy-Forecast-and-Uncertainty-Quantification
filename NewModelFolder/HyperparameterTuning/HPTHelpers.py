import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import os
import optuna
from optuna.trial import Trial
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LSTMModel import Config, LSTMForecast

def train_model(config, train_loader, val_loader, train_size, val_size, device, 
                trial=None, max_epochs=None):

    model = LSTMForecast(config).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Early stopping
    patience = 10
    best_val_loss = np.inf
    epochs_no_improve = 0
    best_model_state = None
    
    epochs = max_epochs if max_epochs is not None else config.epochs
    
    for epoch in range(1, epochs + 1):

        # Training
        model.train()
        epoch_loss = 0
        
        for enc, dec, tgt in train_loader:
            enc, dec, tgt = enc.to(device), dec.to(device), tgt.to(device)
            optimizer.zero_grad()
            output,_ = model(enc, dec)
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
                output,_ = model(enc, dec)
                val_loss_epoch += criterion(output, tgt).item() * enc.size(0)
        val_loss = val_loss_epoch / val_size
        
        scheduler.step(val_loss)
        
        # Report to Optuna for pruning (if trial is provided)
        if trial is not None:
            trial.report(val_loss, epoch)
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return best_val_loss, model

def trialSuggestions(trial: Trial, train_dataset, val_dataset, device, 
              local=False, logger=None):
    
    # Create config with suggested hyperparameters
    config = Config()
    
    # Suggest hyperparameters
    config.hidden_size = trial.suggest_int('hidden_size', 64, 256, step=32)
    config.num_layers = trial.suggest_int('num_layers', 1, 4)
    config.dropout = trial.suggest_float('dropout', 0.1, 0.5)
    config.batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    config.learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    config.device = device
    
    logger.info(f"Trial {trial.number} starting with hyperparameters:\n"
                f"                                                             \033[1mhidden size     :\033[0m\033[37m {config.hidden_size}\n"
                f"                                                             \033[1mnumber of layers:\033[0m\033[37m {config.num_layers}\n"
                f"                                                             \033[1mdropout         :\033[0m\033[37m {config.dropout:.6f}\n"
                f"                                                             \033[1mbatch size      :\033[0m\033[37m {config.batch_size}\n"
                f"                                                             \033[1mlearning rate   :\033[0m\033[37m {config.learning_rate:.6f}")
    
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
            device, trial=trial, max_epochs=1
        )
        
        return best_val_loss
    
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")


def load_dataset(local=False, filePaths=None, logger=None):
    
    # Setup paths
    dataset_path = filePaths[0]

    dataset = torch.load(dataset_path, weights_only=True)
    encoder_data = dataset['encoder']
    decoder_data = dataset['decoder']
    target_data = dataset['target']
    full_dataset = TensorDataset(encoder_data, decoder_data, target_data)

    logger.info("Dataset loaded")

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
    
    logger.info(f"Dataset split into \033[1m(train | val | test)\033[0m\033[37m with sizes: \033[1m({train_size} | {val_size} | {test_size})\033[0m\033[37m")
    
    return train_dataset, val_dataset, test_dataset

