import argparse
from datetime import datetime
import optuna

from HyperparameterTuning.HPTHelpers import (
    run_hyperparameter_search
)
from HyperparameterTuning import HPTOutput as output


def hptmain(n_trials, local, verbose):
    """Main entry point for hyperparameter tuning and analysis."""
    # Run hyperparameter search
    study = run_hyperparameter_search(
        n_trials=n_trials,
        local=local,
        verbose=verbose
    )

    return study