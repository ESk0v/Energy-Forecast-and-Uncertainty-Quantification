import argparse
import os
import time

from Data.DatasetCreation import main as create_dataset
from HyperparameterTuning.HPTMain import hptmain
from LSTM.LSTMMain import LSTMMain as train_model
from Ensemble.EnsembleMain import main as EnsembleModel
from Logger import setup_logger
from LSTMModel import Config

# ==========================================================
# CONFIG
# ==========================================================

SERVER_DATASET_PATH = "/ceph/project/SW6-Group18-Abvaerk/NewModelFolder/Files/dataset.pt"
LOCAL_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Files", "dataset.pt")

SERVER_JSON_FOR_HPT_PATH = "/ceph/project/SW6-Group18-Abvaerk/NewModelFolder/Files/HPTTuning.json"
LOCAL_JSON_FOR_HPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Files", "HPTTuning.json")

SERVER_RINGKØBING_PATH = "/ceph/project/SW6-Group18-Abvaerk/NewModelFolder/Files/RingkøbingData.csv"
LOCAL_RINGKØBING_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Files", "RingkøbingData.csv")

SERVER_MODELDIR_PATH = "/ceph/project/SW6-Group18-Abvaerk/NewModelFolder/Models"
LOCAL_MODELDIR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Models")

# ==========================================================
# DATASET CHECK
# ==========================================================

def getModelPath(model_dir):

    os.makedirs(model_dir, exist_ok=True)

    existing = [f for f in os.listdir(model_dir) if f.startswith("model_v")]
    existing_versions = []
    for f in existing:
        try:
            v = int(f.replace("model_v", ""))
            existing_versions.append(v)
        except ValueError:
            pass
    next_version = max(existing_versions, default=0) + 1
    model_path = os.path.join(model_dir, f"model_v{next_version}")
    os.makedirs(model_path, exist_ok=True)
    model_save_path = os.path.join(model_path, f"model_v{next_version}.pth")

    return model_save_path, next_version

def ensure_dataset_exists(local=False, logger=None):

    logger.info("Checking dataset existence...")

    filePaths = [
            LOCAL_RINGKØBING_PATH if local else SERVER_RINGKØBING_PATH,
            LOCAL_DATASET_PATH if local else SERVER_DATASET_PATH
        ]

    if not os.path.exists(filePaths[1]):
        logger.info("Dataset not found, Creating a new dataset")

        create_dataset(local=local, filePaths=filePaths, logger=logger)

        if os.path.exists(filePaths[1]):
            logger.success("Dataset created successfully")
            return
        else:
            logger.error("Failed to create dataset")
            raise FileNotFoundError(f"Dataset not found at {filePaths[1]} after creation attempt.")

    logger.info("Dataset exists")

def RunTuning(local=False, n_trials=50, epochs=1, tune_patience=1, logger=None):
    #Start the Tuning part

    filePaths = [
        LOCAL_DATASET_PATH if local else SERVER_DATASET_PATH,
        LOCAL_JSON_FOR_HPT_PATH if local else SERVER_JSON_FOR_HPT_PATH
    ]
    #Check if the dataset exist
    logger.info("Starting hyperparameter tuning...")
    #Run the HyperparameterTuning
    study = hptmain(
        n_trials=n_trials,
        epochs=epochs,
        patience = tune_patience,
        local=local,
        filePaths=filePaths,
        logger=logger
    )

    logger.success("Hyperparameter Tuning complete!")

def RunLstm(local=False, epochs=1, train_patience=1, logger=None):
    logger.info("Starting LSTM training...")
    
    modelPath, version = getModelPath(LOCAL_MODELDIR_PATH if local else SERVER_MODELDIR_PATH)
    logger.info(f"Model will be saved as model_v{version}.pth")
    filePaths = [
        LOCAL_DATASET_PATH if local else SERVER_DATASET_PATH,
        modelPath
    ]

    train_model(filePaths=filePaths, epochs=epochs, patience=train_patience, logger=logger)
    logger.success("Finished LSTM plotting")

def RunEnsemble(local=False, epochs=1, n_models=3, ensemble_patience=1, logger=None):
    logger.info("Starting ensemble...")

    # Always find the latest model_vN/ folder automatically
    model_dir = LOCAL_MODELDIR_PATH if local else SERVER_MODELDIR_PATH
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    existing = [f for f in os.listdir(model_dir)
                if os.path.isdir(os.path.join(model_dir, f))
                and f.startswith("model_v")]
    versions = [int(f.replace("model_v", "")) for f in existing
                if f.replace("model_v", "").isdigit()]
    if not versions:
        raise FileNotFoundError(f"No versioned run folders found in {model_dir}")
    latest_run_dir = os.path.join(model_dir, f"model_v{max(versions)}")

    filePaths = [
        LOCAL_DATASET_PATH if local else SERVER_DATASET_PATH,
        model_dir,
        latest_run_dir,
    ]

    EnsembleModel(filePaths=filePaths, epochs=epochs, n_models=n_models, patience=ensemble_patience, logger=logger)
    logger.success("Finished Ensemble plotting")

def Main():
    parser = argparse.ArgumentParser(description="LSTM Pipeline Controller")

    # --mode tune      → run only hyperparameter tuning (Optuna)
    # --mode train     → run only LSTM training + evaluation plots
    # --mode ensemble  → run only the ensemble model
    # --mode full      → run tuning → training → ensemble in sequence
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["tune", "train", "ensemble", "full"],
        help="Which part of the pipeline to run"
    )

    # --local          → use relative local paths instead of server paths (for local testing)
    parser.add_argument("--local", action="store_true", help="Use local paths")

    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials for tuning")
    # --n_models <int> → number of models to train for the ensemble (default: 5)
    parser.add_argument("--n_models", type=int, default=3, help="Number of models for ensemble")
    # --tune_epochs <int> → number of epochs to train during tuning (default: 1)
    parser.add_argument("--tune_epochs", type=int, default=1, help="Number of epochs for tuning")
    # --train_epochs <int> → number of epochs to train during final training (default: 1)
    parser.add_argument("--train_epochs", type=int, default=1, help="Number of epochs for training")
    # --ensemble_epochs <int> → number of epochs to train each ensemble model (default: 1)
    parser.add_argument("--ensemble_epochs", type=int, default=1, help="Number of epochs for ensemble training")
    # --tune_patience <int> → early stopping patience (in epochs) used during hyperparameter tuning (default: 1)
    parser.add_argument("--tune_patience", type=int, default=1, help="Early-stopping patience (epochs) during tuning")
    # --train_patience <int> → early stopping patience (in epochs) used during final LSTM training (default: 1)
    parser.add_argument("--train_patience", type=int, default=1, help="Early-stopping patience (epochs) during training")
    # --ensemble_patience <int> → early stopping patience (in epochs) used when training ensemble members (default: 1)
    parser.add_argument("--ensemble_patience", type=int, default=1, help="Early-stopping patience (epochs) during ensemble training")

    args = parser.parse_args()

    # Initialize logger
    mainLogger = setup_logger("Main", local=args.local)
    trainLogger = setup_logger("LSTM", local=args.local)
    ensembleLogger = setup_logger("Ensemble", local=args.local)
    HPTLogger = setup_logger("Hyperparameter Tuning", local=args.local)
    datasetLogger = setup_logger("Dataset", local=args.local)

    ensure_dataset_exists(local=args.local, logger=datasetLogger)

    Config.print_config(mainLogger)

    if args.mode == "tune":
        RunTuning(
            local=args.local,
            n_trials=args.n_trials,
            epochs=args.tune_epochs,
            tune_patience=args.tune_patience,
            logger=HPTLogger
        )

    elif args.mode == "train":
        RunLstm(
            local=args.local,
            epochs=args.train_epochs,
            train_patience=args.train_patience,
            logger=trainLogger
        )

    elif args.mode == "ensemble":
        # Run ensemble on the latest (or only) existing model_vN/ folder
        RunEnsemble(
            local=args.local, 
            epochs=args.ensemble_epochs, 
            n_models=args.n_models,
            ensemble_patience=args.ensemble_patience, 
            logger=ensembleLogger
        )

    elif args.mode == "full":
        start = time.perf_counter()
        RunTuning(
            local=args.local,
            n_trials=args.n_trials,
            epochs=args.tune_epochs,
            tune_patience=args.tune_patience,
            logger=HPTLogger
        )
        tuning_time = time.perf_counter() - start
        mainLogger.debug(f"Tuning completed in {tuning_time:.2f} seconds")

        start = time.perf_counter()
        RunLstm(
            local=args.local,
            epochs=args.train_epochs,
            train_patience=args.train_patience,
            logger=trainLogger
        )
        train_time = time.perf_counter() - start
        mainLogger.debug(f"LSTM training completed in {train_time:.2f} seconds")

        start = time.perf_counter()
        RunEnsemble(
            local=args.local, 
            epochs=args.ensemble_epochs, 
            n_models=args.n_models,
            ensemble_patience=args.ensemble_patience, 
            logger=ensembleLogger
        )
        ensemble_time = time.perf_counter() - start
        mainLogger.debug(f"Ensemble training completed in {ensemble_time:.2f} seconds")
        
if __name__ == "__main__":
    Main()