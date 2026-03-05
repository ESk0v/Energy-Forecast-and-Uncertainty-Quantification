import os
import sys
from pathlib import Path

from .EnsembleHelpers import _DataLoader, _DatasetSplit, _TrainEnsemble, _LoadEnsembleModels
from .EnsembleOutput import _EvaluateModel

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from LSTMModel import Config

def main(filePaths=None, epochs=1, n_models=3):
    """
    Run the ensemble pipeline.

    Args:
        filePaths: List of [dataset_path, model_dir, run_dir].
                   run_dir is the versioned model folder
                   (e.g. Models/model_v1/).
                   Ensemble models are saved to run_dir/EnsembleModel/.
                   Ensemble plots are saved to run_dir/Plots/.
    """

    # -----------------------------
    # Paths
    # -----------------------------
    dataset_path      = Path(filePaths[0])
    run_dir           = Path(filePaths[2])   # e.g. .../Models/model_v1/
    plot_dir          = run_dir / "Plots"
    ensemble_save_dir = run_dir / "EnsembleModel"

    plot_dir.mkdir(parents=True, exist_ok=True)
    ensemble_save_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = Config()
    batch_size = config.batch_size
    device = config.device

    # Load dataset
    dataset, demand_mean, demand_std = _DataLoader(dataset_path, ensemble_save_dir)
    trainLoader, valLoader, testLoader = _DatasetSplit(dataset, batch_size)

    # Train Ensemble
    print("=== Training ensemble ===")
    _TrainEnsemble(
        n_models=n_models,
        epochs=epochs,
        train_loader=trainLoader,
        val_loader=valLoader,
        config=config,
        save_dir=ensemble_save_dir
    )
    print("Ensemble training complete.")

    # Load Ensemble Models
    models = _LoadEnsembleModels(ensemble_save_dir, config)
    print(f"Loaded {len(models)} ensemble models.")

    _EvaluateModel(testLoader, models, device, demand_mean, demand_std, n_models, plot_dir=plot_dir)
