from torch.utils.data import DataLoader
from LSTMModel import Config
from LSTM.Plotting import main as generate_plots
from LSTM.LSTMTraining import (
    load_and_split_dataset,
    train_model
)


def LSTMMain(filePaths=None, logger=None):    
    # Paths
    dataset_path = filePaths[0]
    model_save_path = filePaths[1]
    
    # Load and split dataset
    train_dataset, val_dataset, train_size, val_size = load_and_split_dataset(dataset_path)
    logger.info(f"Dataset loaded: {train_size} training samples, {val_size} validation samples")
    
    # Create config and data loaders
    config = Config()
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Train model
    best_val_loss, train_losses, val_losses = train_model(
        config, train_loader, val_loader, train_size, val_size, 
        model_save_path, logger
    )

    generate_plots(train_losses, val_losses, model_save_path, logger)