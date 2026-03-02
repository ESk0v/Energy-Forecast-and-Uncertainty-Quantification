from pathlib import Path
import torch

# -----------------------------
# Project root
# -----------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent

# -----------------------------
# Paths
# -----------------------------
DATASET_PATH = PROJECT_ROOT / "NewModelFolder" / "Files" / "dataset.pt"
ENSEMBLE_SAVE_DIR = PROJECT_ROOT / "NewModelFolder" / "Models" / "EnsembleModel"
PLOT_DIR = PROJECT_ROOT / "NewModelFolder" / "Plots"

# Ensure directories exist
ENSEMBLE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Device
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Training config
# -----------------------------
BATCH_SIZE = 64
ENSEMBLE_SIZE = 1
EPOCHS = 1