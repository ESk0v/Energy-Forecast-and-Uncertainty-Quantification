
import os
import sys

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from Data.DatasetCreation import main as create_dataset
from LSTM.LSTMTraining import main as train_model
from LSTM.Plotting import main as generate_plots

# This function handle the main
def LSTMMain(local=False):
    print(f"{'='*60} LSTM Forecast Pipeline {'='*60}\n")
    print(f"[Step 1/4] Dataset Creation")

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

    print(f"[Step 2/3] LSTM Training")
    print("-" * 40)
    train_model(local=local)
    print()

    print(f"[Step 3/4] Evaluation Plotting")
    print("-" * 40)
    generate_plots(local=local)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete.")
    print(f"{'='*60}")