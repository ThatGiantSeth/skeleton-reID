import torch
import numpy as np
import CNN as cnn
import time

window = 20
joints = 15
num_classes = 4  # Updated to match the number of persons
persons = ['antonio', 'aubrey', 'noor', 'seth']

def compute_normalization_stats(skeletons, center_root=True, root_joint=0, eps=1e-6):
    """
    Compute normalization statistics (mean and std) from skeleton data.
    This must be computed from training data to match train.py normalization.
    """
    channel_sums = np.zeros(3, dtype=np.float64)
    channel_sq_sums = np.zeros(3, dtype=np.float64)
    total = 0

    for data in skeletons:
        if center_root:
            root = data[:, root_joint, :][:, None, :]
            data = data - root

        flat = data.reshape(-1, data.shape[2])  # (frames*joints, channels)
        channel_sums += flat.sum(axis=0)
        channel_sq_sums += np.square(flat).sum(axis=0)
        total += flat.shape[0]

    mean = channel_sums / total
    var = channel_sq_sums / total - mean ** 2
    std = np.sqrt(np.maximum(var, eps))
    return mean.astype(np.float32), std.astype(np.float32)

def classifier_model():
    model = cnn.CNNet(window_size=window, num_joints=joints, num_class=num_classes, drop_prob=0.5)
    return model

def normalize_skeleton_data(data, mean, std, center_root=True, root_joint=0):
    """
    Apply root centering and normalization to skeleton data.
    """
    if center_root:
        root = data[:, root_joint, :][:, None, :]
        data = data - root
    
    mean_bc = mean.reshape(1, 1, -1)
    std_bc = np.maximum(std.reshape(1, 1, -1), 1e-6)
    data = (data - mean_bc) / std_bc
    return data

def identify_person(numpy_array, mean, std):
    """
    Accepts a numpy array of shape (20, 15, 3) representing skeleton data for 20 frames, 15 joints, 3 coordinates.
    Returns the identified person name.
    """
    if numpy_array.shape != (20, 15, 3):
        raise ValueError("Input numpy array must have shape (20, 15, 3)")

    # Apply normalization
    numpy_array = normalize_skeleton_data(numpy_array, mean, std)
    
    tensor = torch.from_numpy(numpy_array).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, 10, 15)
    
    model.eval()
    with torch.no_grad():
        logits, _ = model(tensor)
        pred = torch.argmax(logits, dim=1).item()
        return persons[pred]

# Load the model
model = classifier_model()
model.load_state_dict(torch.load('../training/skeleton_model_best.pth', map_location='cpu'))

# Compute normalization statistics from training data
print("Computing normalization statistics from training data...")
from pathlib import Path
import re

def get_arrays(directory="./data", trim_front=499):
    arrays = []
    labels = []
    people = {}
    directory = Path(directory)
    npy_files = sorted(directory.glob("*.npy"))

    for file in npy_files:
        match = re.match(r'([a-zA-Z\-\']+)_.*\.npy', file.name)
        if not match:
            continue
        
        person = match.group(1)
        if person not in people:
            people[person] = len(people)

        array = np.load(file)
        if trim_front > 0:
            array = array[trim_front:]
        
        arrays.append(array)
    return arrays

# Load training data to compute normalization stats
train_arrays = get_arrays("../training/data")
mean, std = compute_normalization_stats(train_arrays)
print(f"Normalization stats computed. Mean shape: {mean.shape}, Std shape: {std.shape}")

# Example usage with data from data_val
# Load a sample from validation data
sample_data = np.load('../training/data_val/noor_standing1000.npy')
# Select a random window of 20 frames
max_start_idx = sample_data.shape[0] - window
random_start_idx = np.random.randint(0, max_start_idx)
input_array = sample_data[random_start_idx:random_start_idx + window]
print(f"Using frames {random_start_idx} to {random_start_idx + window - 1}")

start_time = time.process_time()
person = identify_person(input_array, mean, std)
end_time = time.process_time()

inference_time = end_time - start_time

print(f"Identified person: {person}")
print("Inference time:", inference_time)
print("Expected person: noor")

