from pathlib import Path

from .EnsembleHelpers import _DataLoader, _DatasetSplit, _TrainEnsemble, _LoadEnsembleModels
from .EnsembleOutput import _EvaluateModel
from .EnsembleConfig import DEVICE, ENSEMBLE_SIZE

def main(filePaths=None):

    # -----------------------------
    # Paths
    # -----------------------------
    dataset_path      = filePaths[0]
    run_dir           = filePaths[2]
    plot_dir          = run_dir / "Plots"
    ensemble_save_dir = run_dir / "EnsembleModel"

    plot_dir.mkdir(parents=True, exist_ok=True)
    ensemble_save_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset, demand_mean, demand_std = _DataLoader(dataset_path)
    trainLoader, valLoader, testLoader = _DatasetSplit(dataset)

    # Train Ensemble
    print("=== Training ensemble ===")
    _TrainEnsemble(
        n_models=ENSEMBLE_SIZE,
        train_loader=trainLoader,
        val_loader=valLoader,
        device=DEVICE,
        save_dir=ensemble_save_dir
    )
    print("Ensemble training complete.")

    # Load Ensemble Models
    models = _LoadEnsembleModels(ensemble_save_dir, DEVICE)
    print(f"Loaded {len(models)} ensemble models.")

    _EvaluateModel(testLoader, models, DEVICE, demand_mean, demand_std, plot_dir=plot_dir)
