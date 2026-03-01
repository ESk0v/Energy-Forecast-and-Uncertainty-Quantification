import argparse
import sys

from HyperparameterTuning.HyperparameterTuning import run_hyperparameter_search
from LSTM.Main import main as train_model

def RunTuning(local=False, n_trials=50, dataset_path=None, verbose=False):
    print("Starting hyperparameter tuning...")

    study = run_hyperparameter_search(
        n_trials=n_trials,
        local=local,
        dataset_path=dataset_path,
        verbose=verbose
    )

    print("Finished hyperparameter tuning.")
    return study  # Optional: return study object if needed


def RunLstm(local=False):
    print("Starting LSTM training...")
    train_model(local=local)  # actually runs the training
    print("Finished LSTM training.")


def RunEnsemble():
    print("Starting ensemble...")
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
        RunTuning()
        RunLstm()
        RunEnsemble()


if __name__ == "__main__":
    Main()