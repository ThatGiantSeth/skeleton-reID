## Recording Data

## Training Script
The training script `network/train.py` takes in a set of NumPy arrays containing skeleton data, normalizes them, and trains a CNN `network/CNN.py` on this data. It outputs a folder `fold_models/` containing the best model from each validation fold (see [section that doesn't exist yet]).
### Input data
The CNN is set up to accept skeleton data consisting of 3D world coordinates from an OpenNI-compatible camera like PrimeSense Carmine 1.09 or Microsoft Kinect 360. Any OpenNI camera will work, but the same camera should be used for both training and demonstration for coordinate system consistency. 
The script searches for `.npy` files in a directory called `data`, located within the script's directory (`./network` by default). The files must be named `<person-name>_xxxxxxxx.npy`. If the files were recorded using the provided recording script, they should already have this format. Anything after the first underscore is ignored.

> [!warning] 
> The training script currently acquires its classes from the unique names associated with the files so that multiple samples can be provided per person (e.g. `person1_standing`, `person1_walking`). Therefore, it is probably a good idea to use more than just first names like we do in our example.

## Client

## Server