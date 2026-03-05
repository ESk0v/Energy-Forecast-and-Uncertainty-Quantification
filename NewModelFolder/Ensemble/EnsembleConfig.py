import torch

# -----------------------------
# Device
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Training config
# -----------------------------
BATCH_SIZE = 64
ENSEMBLE_SIZE = 3
EPOCHS = 1
