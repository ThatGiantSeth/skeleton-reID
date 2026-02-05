from pathlib import Path
import re
import numpy as np


def get_arrays(directory="./data", trim_front=499):
    arrays = []
    labels = []
    people = {}
    directory = Path(directory)
    npy_files = sorted(directory.glob("*.npy"))

    for file in npy_files:
        # get files named person_*****.npy
        match = re.match(r"([a-zA-Z\-\']+)_.*\.npy", file.name)
        if not match:
            continue

        person = match.group(1)

        # handle multiple recordings per person
        if person not in people:
            people[person] = len(people)

        array = np.load(file)

        # trim the specified # of frames from the array to remove calibration period
        if trim_front > 0:
            array = array[trim_front:]

        label = people[person]
        arrays.append(array)
        labels.append(label)

    print(f"Loaded {len(arrays)} arrays for {len(people)} people from {directory}")
    print(f"People + labels: {people}")
    return arrays, labels, people


def normalize_skeleton(data, root_joint=0, eps=1e-6):
    # Center data around skeleton instead of world coords
    root = data[:, root_joint, :][:, None, :]
    data = data - root

    # Compute per-channel min/max for normalization
    flat = data.reshape(-1, data.shape[2])  # (frames*joints, channels)
    channel_mins = flat.min(axis=0)
    channel_maxs = flat.max(axis=0)
    channel_range = np.maximum(channel_maxs - channel_mins, eps)

    # Normalize x,y,z to [0,1]
    min_bc = channel_mins.reshape(1, 1, -1)
    range_bc = channel_range.reshape(1, 1, -1)
    data = (data - min_bc) / range_bc

    return data
