from pathlib import Path
import os
from .EnsembleHelpers import _DataLoader, _DatasetSplit, _TrainEnsemble, _LoadEnsembleModels
from .EnsembleOutput import _EvaluateModel
from .EnsembleConfig import DEVICE, ENSEMBLE_SIZE

def main(filePaths=None, logger=None):

    # -----------------------------
    # Paths
    # -----------------------------
    dataset_path      = filePaths[0]
    run_dir           = filePaths[2]
    plot_dir          = os.path.join(run_dir, "Plots")
    ensemble_save_dir = os.path.join(run_dir, "EnsembleModels")

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(ensemble_save_dir, exist_ok=True)

    # Load dataset
    dataset, demand_mean, demand_std = _DataLoader(dataset_path)
    trainLoader, valLoader, testLoader = _DatasetSplit(dataset)

    # Train Ensemble
    _TrainEnsemble(
        n_models=ENSEMBLE_SIZE,
        train_loader=trainLoader,
        val_loader=valLoader,
        device=DEVICE,
        save_dir=ensemble_save_dir,
        logger=logger
    )
    logger.success("Ensemble training completed successfully!")

    # Load Ensemble Models
    models = _LoadEnsembleModels(ensemble_save_dir, DEVICE)

    logger.info("Generating Ensemble plots...")

    _EvaluateModel(testLoader, models, DEVICE, demand_mean, demand_std, plot_dir=plot_dir)
