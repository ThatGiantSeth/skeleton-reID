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


def combine_recordings(directory="./data", trim_front=499, people_map=None):
    directory = Path(directory)
    npy_files = sorted(directory.glob("*.npy"))

    people = dict(people_map) if people_map is not None else {}
    frame_arrays = []
    frame_labels = []

    for file in npy_files:
        match = re.match(r"([a-zA-Z\-\']+)_.*\.npy", file.name)
        if not match:
            continue

        person = match.group(1)

        if person not in people:
            if people_map is not None:
                raise ValueError(f"Unknown person '{person}' found in {file.name} for provided people_map")
            people[person] = len(people)

        label = people[person]
        array = np.load(file)

        if trim_front > 0:
            array = array[trim_front:]

        if array.shape[0] == 0:
            continue

        frame_arrays.append(array)
        frame_labels.append(np.full(array.shape[0], label, dtype=np.int64))

    if not frame_arrays:
        raise ValueError(f"No valid .npy recordings found in {directory}")

    x = np.concatenate(frame_arrays, axis=0)
    y = np.concatenate(frame_labels, axis=0)

    unique_labels, counts = np.unique(y, return_counts=True)
    label_counts = {int(label): int(count) for label, count in zip(unique_labels, counts)}

    print(f"Loaded {len(frame_arrays)} recordings for {len(people)} people from {directory}")
    print(f"Combined frames shape: {x.shape}")
    print(f"Frame counts per label: {label_counts}")
    print(f"People + labels: {people}")

    return x, y, people


def normalize_skeleton(data, root_joint=0, eps=1e-6):
    #center data around skeleton instead of physical space
    root = data[:, root_joint, :][:, None, :]
    data = data - root

    # get min/max for x,y,z
    flat = data.reshape(-1, data.shape[2])
    channel_mins = flat.min(axis=0)
    channel_maxs = flat.max(axis=0)
    channel_range = np.maximum(channel_maxs - channel_mins, eps)

    # normalize x,y,z to [0,1]
    min_bc = channel_mins.reshape(1, 1, -1)
    range_bc = channel_range.reshape(1, 1, -1)
    data = (data - min_bc) / range_bc

    return data


def window_sequence(x, y, window_size, stride):
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of frames")

    windows = []
    window_labels = []

    for window_start in range(0, x.shape[0] - window_size + 1, stride):
        x_window = x[window_start:window_start + window_size]
        y_window = y[window_start:window_start + window_size]

        if np.any(y_window != y_window[0]):
            continue

        windows.append(x_window)
        window_labels.append(int(y_window[0]))

    if not windows:
        raise ValueError("No windows were created.")

    windows = np.stack(windows, axis=0)
    window_labels = np.asarray(window_labels, dtype=np.int64)

    print(f"Created {windows.shape[0]} windows of shape {windows.shape[1:]}")
    return windows, window_labels
