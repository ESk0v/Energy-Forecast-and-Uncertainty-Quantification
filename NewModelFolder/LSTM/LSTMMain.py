
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from LSTM.LSTMTraining import main as train_model
from LSTM.Plotting import main as generate_plots

# This function handle the main
def LSTMMain(local=False, filePaths=None):

    print(f"[Step 2/3] LSTM Training")
    print("-" * 40)
    train_model(local=local, filePaths=filePaths)
    print()

    print(f"[Step 3/4] Evaluation Plotting")
    print("-" * 40)
    generate_plots(local=local)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete.")
    print(f"{'='*60}")