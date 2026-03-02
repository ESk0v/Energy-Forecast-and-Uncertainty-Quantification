from pathlib import Path
import torch

from .EnsembleHelpers import _DataLoader, _DatasetSplit, _TrainEnsemble, _LoadEnsembleModels
from .EnsembleOutput import _EvaluateModel
from .EnsembleConfig import DEVICE, ENSEMBLE_SIZE, ENSEMBLE_SAVE_DIR


#This function runs the ensemble model that has inherent model uncertainty
def main():
    #

    #Load dataset
    dataset = _DataLoader()

    trainLoader, valLoader, testLoader = _DatasetSplit(dataset)

    # Train Ensemble
    print("=== Training ensemble ===")
    _TrainEnsemble(
        n_models=ENSEMBLE_SIZE,
        train_loader=trainLoader,
        val_loader=valLoader,
        device=DEVICE,
        save_dir=ENSEMBLE_SAVE_DIR
    )
    print("Ensemble training complete.")

    # Load Ensemble Models
    models = _LoadEnsembleModels(ENSEMBLE_SAVE_DIR, DEVICE)
    print(f"Loaded {len(models)} ensemble models.")

    _EvaluateModel(testLoader, models, DEVICE)