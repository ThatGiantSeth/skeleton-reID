# Getting Started

## Python Install Manager and creating Python virtual environments
> [!important] 
> Because this project requires specific Python versions for different components, it is **highly recommended** to use virtual environments to compartmentalize your client/server/training packages and versions. We will be using example environment names throughout this guide that match their associated script e.g. `client-env`, `server-env`, and `training-env`. If you do not want to use virtual environments you can skip this section, but the rest of this guide will be more confusing.

## Recording script

## Training Script
The training script `network/train.py` takes in a set of NumPy arrays containing skeleton data, normalizes them, and trains a CNN `network/CNN.py` on this data.
### Input data
The CNN is set up to accept skeleton data consisting of 3D world coordinates from an OpenNI-compatible camera like PrimeSense Carmine 1.09 or Microsoft Kinect 360. Any OpenNI camera will work, but the same camera should be used for both training and demonstration for coordinate system consistency. 
The script searches for `.npy` files in a directory called `data`, located within the script's directory (`./network` by default). The files must be named `<person-name>_xxxxxxxx.npy`. If the files were recorded using the provided recording script, they should already have this format. Anything after the first underscore is ignored.

> [!warning] 
> The training script currently acquires its classes from the unique names associated with the files so that multiple samples can be provided per person (e.g. `person1_standing`, `person1_walking`). Therefore, it is probably a good idea to use more than just first names like we do in our example.

### Setting up the environment
The training script requires any of the Python versions supported by PyTorch. As of this documentation, these are **Python 3.11 and newer**. We used Python 3.11 during our project.


## Server

## Client

remember to explain install of opencv including VS 2022 and Cmake 3.4.3