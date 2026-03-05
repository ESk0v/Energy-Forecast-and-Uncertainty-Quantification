import torch

from .EnsembleHelpers import _DataLoader, _DatasetSplit, _TrainEnsemble, _LoadEnsembleModels
from .EnsembleOutput import _EvaluateModel
from .EnsembleConfig import DEVICE, ENSEMBLE_SIZE


#This function runs the ensemble model that has inherent model uncertainty
def main(local=False, filePaths=None):
    #

    # Unpack file paths
    dataset_path = filePaths[0]
    model_dir = filePaths[1]
    plot_dir = filePaths[2]

    #Load dataset
    dataset, demand_mean, demand_std = _DataLoader(dataset_path, model_dir)
    trainLoader, valLoader, testLoader = _DatasetSplit(dataset)

    # Train Ensemble
    print("=== Training ensemble ===")
    _TrainEnsemble(
        n_models=ENSEMBLE_SIZE,
        train_loader=trainLoader,
        val_loader=valLoader,
        device=DEVICE,
        save_dir=model_dir
    )
    print("Ensemble training complete.")

    # Load Ensemble Models
    models = _LoadEnsembleModels(model_dir, DEVICE)
    print(f"Loaded {len(models)} ensemble models.")

    _EvaluateModel(testLoader, models, DEVICE, demand_mean, demand_std, plot_dir)