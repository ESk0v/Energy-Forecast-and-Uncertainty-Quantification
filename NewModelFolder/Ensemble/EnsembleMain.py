import os
import sys

from .EnsembleHelpers import _DataLoader, _DatasetSplit, _TrainEnsemble, _LoadEnsembleModels
from .EnsembleOutput import _EvaluateModel

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from LSTMModel import Config


#This function runs the ensemble model that has inherent model uncertainty
def main(local=False, filePaths=None, epochs=1, n_models=3):
    #

    # Unpack file paths
    dataset_path = filePaths[0]
    model_dir = filePaths[1]
    plot_dir = filePaths[2]

    # Configuration
    config = Config()
    batch_size = config.batch_size
    device = config.device

    # Load dataset
    dataset, demand_mean, demand_std = _DataLoader(dataset_path, model_dir)
    trainLoader, valLoader, testLoader = _DatasetSplit(dataset, batch_size)

    # Train Ensemble
    print("=== Training ensemble ===")
    _TrainEnsemble(
        n_models=n_models,
        epochs=epochs,
        train_loader=trainLoader,
        val_loader=valLoader,
        config=config,
        save_dir=model_dir
    )
    print("Ensemble training complete.")

    # Load Ensemble Models
    models = _LoadEnsembleModels(model_dir, config)
    print(f"Loaded {len(models)} ensemble models.")

    _EvaluateModel(testLoader, models, demand_mean, demand_std, device, plot_dir)