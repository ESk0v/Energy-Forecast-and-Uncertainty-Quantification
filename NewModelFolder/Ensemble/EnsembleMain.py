from pathlib import Path
import torch

from .EnsembleHelpers import _DataLoader, _DatasetSplit, _TrainEnsemble, _LoadEnsembleModels
from .EnsembleOutput import _EvaluateModel
from .EnsembleConfig import DEVICE, ENSEMBLE_SIZE

def main(filePaths=None):
    """
    Run the ensemble pipeline.

    Args:
        filePaths: List of [dataset_path, model_dir, run_dir].
                   run_dir is the versioned single-LSTM folder
                   (e.g. Models/SingleLSTM/model_v1/).
                   Ensemble models are saved to run_dir/Plots/EnsembleModel/.
                   Ensemble plots are saved to run_dir/Plots/.
    """

    # -----------------------------
    # Paths
    # -----------------------------
    dataset_path      = Path(filePaths[0])
    run_dir           = Path(filePaths[2])   # e.g. .../SingleLSTM/model_v1/
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
