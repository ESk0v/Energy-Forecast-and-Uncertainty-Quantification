"""
Main Pipeline Orchestrator
==========================
Executes the full training pipeline in order:

    1. Dataset Creation   — Reads CSV, builds encoder/decoder/target tensors, saves dataset.pt
    2. Dataset Inspection  — (Optional, --inspect) Prints a sample from the dataset for debugging
    3. LSTM Training       — Trains the model, saves the best checkpoint
    4. Evaluation Plotting — Loads checkpoint, runs inference, generates evaluation plots

Usage (standalone):
    python3 Main.py --local                    # Run full pipeline locally
    python3 Main.py --local --inspect          # Run full pipeline locally with dataset inspection
    python3 Main.py                            # Run full pipeline on server (default)
    python3 Main.py --inspect --inspect-idx 5  # Inspect sample at index 5

Usage (via orchestrator):
    Imported and called as main(local=True/False) — no argparse involved.
"""

import os
import sys

# Add the Data folder to sys.path so we can import from the sibling directory
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, "..", "Data"))

from DatasetCreation import main as create_dataset
from DatasetLookup import main as inspect_dataset
from LSTM.LSTMTraining import main as train_model
from LSTM.Plotting import main as generate_plots


def main(local=False, inspect=False, inspect_idx=0):
    mode = "LOCAL" if local else "SERVER"
    print(f"{'='*60}")
    print(f"  LSTM Forecast Pipeline — {mode} mode")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------
    # Step 1: Dataset Creation
    # Only creates a new dataset if dataset.pt does not already exist.
    # -----------------------------------------------------------------
    print(f"[Step 1/4] Dataset Creation")
    print("-" * 40)

    if local:
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.pt")
    else:
        dataset_path = "/ceph/project/SW6-Group18-Abvaerk/ServerReady/dataset.pt"

    if os.path.exists(dataset_path):
        print(f"  Dataset already exists at {dataset_path} — skipping creation.\n")
    else:
        print(f"  No dataset found. Creating...")
        create_dataset(local=local)
        print()

    # -----------------------------------------------------------------
    # Step 2: Dataset Inspection (optional)
    # -----------------------------------------------------------------
    if inspect:
        print(f"[Step 2/4] Dataset Inspection (sample index: {inspect_idx})")
        print("-" * 40)
        inspect_dataset(local=local, line_idx=inspect_idx)
        print()
    else:
        print(f"[Step 2/4] Dataset Inspection — skipped (use --inspect to enable)\n")

    # -----------------------------------------------------------------
    # Step 3: Model Training
    # -----------------------------------------------------------------
    print(f"[Step 3/4] LSTM Training")
    print("-" * 40)
    train_model(local=local)
    print()

    # -----------------------------------------------------------------
    # Step 4: Evaluation Plotting
    # -----------------------------------------------------------------
    print(f"[Step 4/4] Evaluation Plotting")
    print("-" * 40)
    generate_plots(local=local)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LSTM Forecast Training Pipeline")
    parser.add_argument('--local', action='store_true',
                        help='Use local relative paths instead of server paths')
    parser.add_argument('--inspect', action='store_true',
                        help='Inspect a dataset sample after creation (debugging)')
    parser.add_argument('--inspect-idx', type=int, default=0,
                        help='Sample index to inspect (default: 0)')
    args = parser.parse_args()

    main(
        local=args.local,
        inspect=args.inspect,
        inspect_idx=args.inspect_idx
    )