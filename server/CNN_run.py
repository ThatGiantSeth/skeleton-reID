import torch
import numpy as np
import CNN as cnn
import time

window = 10
joints = 15
num_classes = 4  # Updated to match the number of persons
persons = ['antonio', 'aubrey', 'noor', 'seth']

def classifier_model():
    model = cnn.CNNet(window_size=window, num_joints=joints, num_class=num_classes, drop_prob=0.5)
    return model

def identify_person(numpy_array):
    """
    Accepts a numpy array of shape (10, 15, 3) representing skeleton data for 10 frames, 15 joints, 3 coordinates.
    Returns the identified person name.
    """
    if numpy_array.shape != (10, 15, 3):
        raise ValueError("Input numpy array must have shape (10, 15, 3)")

    tensor = torch.from_numpy(numpy_array).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, 10, 15)
    
    model.eval()
    with torch.no_grad():
        logits, _ = model(tensor)
        pred = torch.argmax(logits, dim=1).item()
        return persons[pred]

# Load the model
model = classifier_model()
model.load_state_dict(torch.load('../training/skeleton_model.pth', map_location='cpu'))

# Example usage with data from data_val
# Load a sample from validation data
sample_data = np.load('../training/data_val/noor_standing1000.npy')
# Take the first 10 frames
input_array = sample_data[:10]

start_time = time.process_time()
person = identify_person(input_array)
end_time = time.process_time()

inference_time = end_time - start_time

print(f"Identified person: {person}")
print("Inference time:", inference_time)
print("Expected person: noor")

