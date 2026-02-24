import torch
import numpy as np

# Load your dataset (single .pt file)
data = torch.load("dataset.pt")  # replace with your .pt file path

encoder = data['encoder']
decoder = data['decoder']
target = data['target']

# Function to print full tensor line by line
def print_full_tensor_line(tensor, name, line_idx=0):
    print(f"\n{name} - line {line_idx}:")
    # Convert to numpy for prettier printing
    arr = tensor[line_idx].numpy() if isinstance(tensor[line_idx], torch.Tensor) else tensor[line_idx]
    # Print each row fully
    if arr.ndim == 2:
        for i, row in enumerate(arr):
            print(f"Row {i}: {row}")
    else:
        print(arr)

# Print first encoder sequence
print_full_tensor_line(encoder, "Encoder", line_idx=0)

# Print first decoder sequence
print_full_tensor_line(decoder, "Decoder", line_idx=0)

# Print first target sequence
print_full_tensor_line(target, "Target", line_idx=0)

row = np.array(encoder)  # your row
row2 = np.array(decoder)
row3 = np.array(target)
print("Shape of row:", row.shape)
print("Shape of row:", row2.shape)
print("Shape of row:", row3.shape)