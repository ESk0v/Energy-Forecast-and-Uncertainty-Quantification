import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from LSTM.LSTMTraining import main as train_model
from LSTM.Plotting import main as generate_plots

# This function handle the main
def LSTMMain(local=False, filePaths=None):

    print(f"[Step 2/3] LSTM Training")
    print("-" * 40)
    # train_model returns the per-run folder (e.g. Models/SingleLSTM/model_v1/)
    run_dir = train_model(local=local, filePaths=filePaths)
    print()

    print(f"[Step 3/4] Evaluation Plotting")
    print("-" * 40)
    # Pass dataset_path, model_dir, and the new run_dir as plot_dir
    plot_file_paths = [filePaths[0], filePaths[1], run_dir]
    generate_plots(local=local, filePaths=plot_file_paths)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete.")
    print(f"{'='*60}")