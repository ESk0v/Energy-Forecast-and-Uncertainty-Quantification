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

SERVER_MODELDIR_PATH = "/ceph/project/SW6-Group18-Abvaerk/NewModelFolder/Models/SingleLSTM"
LOCAL_MODELDIR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Models", "SingleLSTM")

SERVER_ENSEMBLE_MODELDIR_PATH = "/ceph/project/SW6-Group18-Abvaerk/NewModelFolder/Models/EnsembleModel"
LOCAL_ENSEMBLE_MODELDIR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Models", "EnsembleModel")

SERVER_PLOTDIR_PATH = "/ceph/project/SW6-Group18-Abvaerk/NewModelFolder/Plots"
LOCAL_PLOTDIR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Plots")

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
    
    train_model(local=local, filePaths=filePaths)
    print("Finished LSTM training.")


def RunEnsemble(local=False):
    print("Starting ensemble...")

    filePaths = [
        LOCAL_DATASET_PATH if local else SERVER_DATASET_PATH,
        LOCAL_ENSEMBLE_MODELDIR_PATH if local else SERVER_ENSEMBLE_MODELDIR_PATH,
        LOCAL_PLOTDIR_PATH if local else SERVER_PLOTDIR_PATH
    ]

    ensure_dataset_exists(local=local, dataset_path=filePaths[0])

    EnsembleModel(local=local, filePaths=filePaths)

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
        RunLstm(
            local=args.local
        )

    elif args.mode == "ensemble":
        RunEnsemble(local=args.local)

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
        RunEnsemble(local=args.local)

if __name__ == "__main__":
    Main()