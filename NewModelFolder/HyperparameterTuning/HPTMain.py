import argparse
from datetime import datetime
import optuna

from HyperparameterTuning.HPTHelpers import (
    run_hyperparameter_search,
    train_with_best_params
)
from HyperparameterTuning import HPTOutput as output


def hptmain(n_trials, local, dataset_path, verbose):
    """Main entry point for hyperparameter tuning and analysis."""
    parser = argparse.ArgumentParser(
        description='LSTM Hyperparameter Tuning with Optuna'
    )
    
    # Mode selection
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Tuning command
    tune_parser = subparsers.add_parser('tune', help='Run hyperparameter tuning')
    tune_parser.add_argument(
        '--local', 
        action='store_true',
        help='Use local relative paths instead of server paths'
    )
    tune_parser.add_argument(
        '--n_trials', 
        type=int, 
        default=50,
        help='Number of trials for hyperparameter search (default: 50)'
    )
    tune_parser.add_argument(
        '--dataset', 
        type=str, 
        default=None,
        help='Path to dataset.pt file (optional)'
    )
    tune_parser.add_argument(
        '--train_final', 
        action='store_true',
        help='Train final model with best parameters after tuning'
    )
    tune_parser.add_argument(
        '--quick', 
        action='store_true',
        help='Quick test run with only 10 trials'
    )
    tune_parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Show detailed training progress (helpful for debugging)'
    )
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'tune':
        # Adjust trials for quick test
        n_trials = 10 if args.quick else args.n_trials
        
        # Run hyperparameter search
        study = run_hyperparameter_search(
            n_trials=n_trials,
            local=args.local,
            dataset_path=args.dataset,
            verbose=args.verbose
        )
        
        # Optionally train final model
        if args.train_final:
            output.print_separator()
            train_with_best_params(study, local=args.local, dataset_path=args.dataset)
        else:
            output.print_train_final_instruction(args.local)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    hptmain()