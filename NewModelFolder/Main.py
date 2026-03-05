import argparse
import os

from Data.DatasetCreation import main as create_dataset
from HyperparameterTuning.HPTMain import hptmain
from LSTM.LSTMMain import LSTMMain as train_model
from Ensemble.EnsembleMain import main as EnsembleModel

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

def ensure_dataset_exists(local=False, dataset_path=None):
    print("Checking dataset existence...")

    if os.path.exists(dataset_path):
        print("Dataset exists.")
        
    else:
        print("Dataset not found, Creating a new dataset")

        filePaths = [
            LOCAL_RINGKØBING_PATH if local else SERVER_RINGKØBING_PATH,
            LOCAL_DATASET_PATH if local else SERVER_DATASET_PATH
        ]

        create_dataset(local=local, filePaths=filePaths)

def RunTuning(local=False, n_trials=50, verbose=False):
    #Start the Tuning part
    print("Starting hyperparameter tuning...")

    filePaths = [
        LOCAL_DATASET_PATH if local else SERVER_DATASET_PATH,
        LOCAL_JSON_FOR_HPT_PATH if local else SERVER_JSON_FOR_HPT_PATH
    ]
    #Check if the dataset exist
    ensure_dataset_exists(local=local, dataset_path=filePaths[0])

    #Run the HyperparameterTuning
    study = hptmain(
        n_trials=n_trials,
        local=local,
        verbose=verbose,
        filePaths=filePaths
    )

    #Hyper ParameterTuning is done
    print("Finished hyperparameter tuning.")


def RunLstm(local=False):
    print("Starting LSTM training...")
    
    filePaths = [
        LOCAL_DATASET_PATH if local else SERVER_DATASET_PATH,
        LOCAL_MODELDIR_PATH if local else SERVER_MODELDIR_PATH,
    ]

    ensure_dataset_exists(local=local, dataset_path=filePaths[0])

    run_dir = train_model(filePaths=filePaths)
    print("Finished LSTM training.")

    return run_dir


def RunEnsemble(local=False, run_dir=None):
    print("Starting ensemble...")

    if run_dir is None:
        # Standalone ensemble run — find the latest model_vN/ folder
        model_dir = LOCAL_MODELDIR_PATH if local else SERVER_MODELDIR_PATH
        existing = [f for f in os.listdir(model_dir)
                    if os.path.isdir(os.path.join(model_dir, f))
                    and f.startswith("model_v")]
        versions = [int(f.replace("model_v", "")) for f in existing
                    if f.replace("model_v", "").isdigit()]
        if not versions:
            raise FileNotFoundError(f"No versioned run folders found in {model_dir}")
        run_dir = os.path.join(model_dir, f"model_v{max(versions)}")

    filePaths = [
        LOCAL_DATASET_PATH if local else SERVER_DATASET_PATH,
        LOCAL_MODELDIR_PATH if local else SERVER_MODELDIR_PATH,
        run_dir,
    ]

    EnsembleModel(filePaths=filePaths)
    print("Finished ensemble.")


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
    # --verbose        → enable verbose logging output during tuning
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    # --n_trials <int> → number of Optuna trials for hyperparameter tuning (default: 50)
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials for tuning")

    args = parser.parse_args()

    if args.mode == "tune":
        RunTuning(
            local=args.local,
            n_trials=args.n_trials,
            verbose=args.verbose
        )

    elif args.mode == "train":
        RunLstm(local=args.local)

    elif args.mode == "ensemble":
        # Run ensemble on the latest (or only) existing model_vN/ folder
        RunEnsemble(local=args.local)

    elif args.mode == "full":
        print("RUNNING TUNING")
        RunTuning(
            local=args.local,
            n_trials=args.n_trials,
            verbose=args.verbose
        )

        print("RUNNING TRAINING")
        run_dir = RunLstm(local=args.local)

        print("RUNNING ENSEMBLE")
        RunEnsemble(local=args.local, run_dir=run_dir)

if __name__ == "__main__":
    Main()