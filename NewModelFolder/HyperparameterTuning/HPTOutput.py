"""
Output formatting functions for hyperparameter tuning results and analysis.
"""
import os


def print_mode_info(local):
    """Print information about running mode."""
    mode = "LOCAL" if local else "SERVER"
    print(f"Running in {mode} mode")


def print_device_info(device):
    """Print device information."""
    print(f"Using device: {device}")


def print_dataset_info(train_size, val_size, test_size):
    """Print dataset split information."""
    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")


def print_tuning_start(n_trials):
    """Print header for hyperparameter search start."""
    print(f"\nStarting hyperparameter search with {n_trials} trials...")
    print("=" * 70)


def print_verbose_mode():
    """Print verbose mode indicator."""
    print("\n(Verbose mode: showing detailed training progress)\n")


def print_trial_info(trial, config):
    """Print information about current trial."""
    print(f"\n  Trial {trial.number}: hidden_size={config.hidden_size}, "
          f"num_layers={config.num_layers}, dropout={config.dropout:.2f}, "
          f"batch_size={config.batch_size}, lr={config.learning_rate:.6f}")


def print_epoch_info(epoch, train_loss, val_loss):
    """Print training progress for an epoch."""
    print(f"    Epoch {epoch}: train_loss={train_loss:.4f}, "
          f"val_loss={val_loss:.4f}")


def print_early_stopping(epoch):
    """Print early stopping message."""
    print(f"    Early stopped at epoch {epoch}")


def print_trial_pruned(epoch):
    """Print trial pruned message."""
    print(f"    Trial pruned at epoch {epoch}")


def print_tuning_complete():
    """Print header for tuning completion."""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 70)


def print_best_trial(trial):
    """Print best trial information."""
    print(f"\nBest trial:")
    print(f"  Validation Loss: {trial.value:.6f}")
    print(f"\n  Best Hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


def print_best_params_saved(best_params_file):
    """Print best parameters file save location."""
    print(f"Best parameters saved to: {best_params_file}")


def print_final_model_header():
    """Print header for final model training."""
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("=" * 70)


def print_final_model_hyperparameters(best_params):
    """Print hyperparameters for final model."""
    print("\nHyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print()


def print_final_model_saved(model_save_path, best_val_loss):
    """Print final model save information."""
    print(f"\nFinal model saved to: {model_save_path}")
    print(f"Final validation loss: {best_val_loss:.6f}")


def print_train_final_instruction(local):
    """Print instruction for training final model."""
    print("\n" + "=" * 70)
    print("To train a final model with the best parameters, run:")
    print(f"  python main.py tune {'--local' if local else ''} --train_final")
    print("=" * 70)


def print_file_not_found(filepath):
    """Print file not found error."""
    print(f"Error: File not found: {filepath}")


def print_trial_failed(error):
    """Print trial failure message."""
    print(f"Trial failed with error: {error}")


def print_loading_dataset(dataset_path):
    """Print dataset loading message."""
    print(f"Loading dataset from {dataset_path}...")


def print_separator():
    """Print a separator line."""
    print("=" * 70)