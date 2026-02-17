import numpy as np
from pathlib import Path
from preprocessing import get_arrays


def split_samples(source_directory="./data", output_directory="./data_split", frames_per_split=100):
    # Create output directory if it doesn't exist
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load arrays and labels
    arrays, labels, people = get_arrays(directory=source_directory, trim_front=500)
    
    # Match labels to people
    label_to_person = {v: k for k, v in people.items()}
    
    person_sample_count = {}
    
    # Process each array
    for array, label in zip(arrays, labels):
        person_name = label_to_person[label]
        
        if person_name not in person_sample_count:
            person_sample_count[person_name] = 0
        
        # Split the array into equal-sized samples
        total_frames = array.shape[0]
        num_splits = total_frames // frames_per_split
        
        for i in range(num_splits):
            start_idx = i * frames_per_split
            end_idx = start_idx + frames_per_split
            
            split_array = array[start_idx:end_idx]
            
            sample_num = person_sample_count[person_name]
            filename = f"{person_name}_sample_{sample_num}.npy"
            filepath = output_path / filename
            
            # Save the split array
            np.save(filepath, split_array)
            print(f"Saved {filename}: {split_array.shape}")
            
            person_sample_count[person_name] += 1
    
    print(f"\nTotal samples created: {sum(person_sample_count.values())}")


if __name__ == "__main__":
    
    # Split validation data
    print("Splitting validation data...")
    split_samples(
        source_directory="./data_val",
        output_directory="./data_val_split",
        frames_per_split=100
    )
    
    # Split training data
    print("Splitting training data...")
    split_samples(
        source_directory="./data",
        output_directory="./data_split",
        frames_per_split=100
    )
