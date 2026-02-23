# HyperParameterTuning/config.py
import pathlib

# ----------------------------
# Dataset
# ----------------------------
DATASET_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / 'Files' / 'dataset.json'

# ----------------------------
# Training
# ----------------------------
BATCH_SIZE = 32
EPOCHS = 50000

# ----------------------------
# Model
# ----------------------------
INPUT_SIZE = 9          # number of features in dataset
DROPOUT = 0.2

# ----------------------------
# Device
# ----------------------------
DEVICE = "cuda"  # or "cpu"

# ----------------------------
# Save paths
# ----------------------------
LRRT_PLOT_PATH = pathlib.Path(__file__).resolve().parent / "LRRT_plot.png"
GRID_SEARCH_PLOT_PATH = pathlib.Path(__file__).resolve().parent / "GridSearch_plot.png"

# ----------------------------
# Additional options
# ----------------------------
NORMALIZE = True         # whether to normalize dataset
TEST_SIZE = 0.30         # fraction for test set (train+val split handled separately)
VAL_SPLIT = 0.50         # fraction of test set used for validation


#=============================================
# Parameters That Already Have ParameterTuning
#=============================================
LEARNING_RATE = 0.004467 # Get Updated Using LTTR (LearningRateTuning.py)
HIDDEN_SIZE = 64
NUM_LAYERS = 2