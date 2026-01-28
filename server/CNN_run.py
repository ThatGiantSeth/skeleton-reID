import torch
import numpy as np
import CNN as cnn
import time

window = 20
joints = 15
num_classes = 4  # Updated to match the number of persons
persons = ['antonio', 'aubrey', 'noor', 'seth']

def classifier_model():
    model = cnn.CNNet(window_size=window, num_joints=joints, num_class=num_classes, drop_prob=0.5)
    return model

def identify_person(numpy_array):
    """
    Accepts a numpy array of shape (20, 15, 3) representing skeleton data for 20 frames, 15 joints, 3 coordinates.
    Returns the identified person name.
    """
    if numpy_array.shape != (20, 15, 3):
        raise ValueError("Input numpy array must have shape (20, 15, 3)")

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

# Example usage with data from data_val
# Load a sample from validation data
sample_data = np.load('../training/data_val/noor_standing1000.npy')
# Select a random window of 20 frames
max_start_idx = sample_data.shape[0] - window
random_start_idx = np.random.randint(0, max_start_idx)
input_array = sample_data[random_start_idx:random_start_idx + window]
print(f"Using frames {random_start_idx} to {random_start_idx + window - 1}")

start_time = time.process_time()
person = identify_person(input_array)
end_time = time.process_time()

inference_time = end_time - start_time

print(f"Identified person: {person}")
print("Inference time:", inference_time)
print("Expected person: noor")

