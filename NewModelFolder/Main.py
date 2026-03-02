import argparse
import os
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

# ==========================================================
# DATASET CHECK
# ==========================================================

def ensure_dataset_exists(dataset_path: str):
    print("Checking dataset existence...")
    print(f"Expected dataset path: {dataset_path}")

    if os.path.exists(dataset_path):
        print("Dataset found.\n")
        return

    print("Dataset NOT found.")

    # Future option:
    # create_dataset(dataset_path)

    raise FileNotFoundError(
        f"Dataset not found at {dataset_path}. "
        f"Please create dataset before training."
    )

def RunTuning(local=False, n_trials=50, verbose=False):
    #Start the Tuning part
    print("Starting hyperparameter tuning...")

    filePaths = [
        LOCAL_DATASET_PATH if local else SERVER_DATASET_PATH,
        LOCAL_JSON_FOR_HPT_PATH if local else SERVER_JSON_FOR_HPT_PATH
    ]
    #Check if the dataset exist
    #ensure_dataset_exists(filePaths)

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
    train_model(local=local)  # actually runs the training
    print("Finished LSTM training.")


def RunEnsemble():
    print("Starting ensemble...")
    EnsembleModel()
    print("Finished ensemble.")


def Main():
    parser = argparse.ArgumentParser(description="LSTM Pipeline Controller")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["tune", "train", "ensemble", "full"],
        help="Which part of the pipeline to run"
    )

    parser.add_argument("--local", action="store_true", help="Use local paths")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials for tuning")

    args = parser.parse_args()

    if args.mode == "tune":
        RunTuning(
            local=args.local,
            n_trials=args.n_trials,
            verbose=args.verbose
        )

    elif args.mode == "train":
        RunLstm(
            local=args.local
        )

    elif args.mode == "ensemble":
        RunEnsemble()

    elif args.mode == "full":
        print("RUNNING TUNING")
        RunTuning(
            local=args.local,
            n_trials=args.n_trials,
            verbose=args.verbose
        )


        print("RUNNING TRAINING")
        RunLstm(local=args.local)

        print("RUNNING ENSEBMLE")
        RunEnsemble()

if __name__ == "__main__":
    Main()