import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from LSTM.LSTMTraining import main as train_model
from LSTM.Plotting import main as generate_plots

# This function handle the main
def LSTMMain(filePaths=None):

    print(f"[Step 1/2] LSTM Training")
    print("-" * 40)
    # train_model returns the per-run folder (e.g. Models/model_v1/)
    run_dir = train_model(filePaths=filePaths)
    print()

    print(f"[Step 2/2] Evaluation Plotting")
    print("-" * 40)
    # Plots and README_Evaluation.md go into run_dir/Plots/
    plot_file_paths = [filePaths[0], filePaths[1], run_dir]
    generate_plots(filePaths=plot_file_paths)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete.")
    print(f"{'='*60}")

    # Return run_dir so Main.py can pass it to the ensemble step
    return run_dir
