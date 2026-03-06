# Recording Data
A simple script is provided to record skeleton data frames for the training and test sets. It is located in the `client` folder simply because it uses many of the same dependencies as the client.

## How to use:
Set up your camera in your recording environment and connect it to your PC. We recommend using the `SimpleViewer` app, which is installed with OpenNI, to view the camera feed and find the optimal height/angle for recording.

Open a CMD/Powershell window and navigate to the client folder:
```sh
cd <project root>\client
```
> [!note]
> The following commands assume that you have set up your virtual environments in the same way as the guide e.g. located within the project root and with the specified names.

From here, you should be able to run the script. Try doing a test run before actually recording any data to make sure that OpenNI/NiTE are installed properly:
```sh
..\client-env\Scripts\python trainingData.py
```

If OpenNI/NiTE are working properly, you should see a prompt to enter the name of the person you are recording. If you see an OpenNI error (ONI.StatusError, etc.), make sure that OpenNI was installed properly and that you copied the NiTE files into the `client` folder.

When you want to actually record training or test data, start the program as above. 

---next write about entering name + pose data

# Training Script
The training script `network/train.py` takes in a set of NumPy arrays containing skeleton data, normalizes them, and trains a CNN `network/CNN.py` on this data. It outputs a folder `fold_models/` containing the best model from each validation fold (see [section that doesn't exist yet]).
## Input data
The CNN is set up to accept skeleton data consisting of 3D world coordinates from an OpenNI-compatible camera like PrimeSense Carmine 1.09 or Microsoft Kinect 360. Any OpenNI camera will work, but the same camera should be used for both training and demonstration for coordinate system consistency. 
The script searches for `.npy` files in a directory called `data`, located within the script's directory (`./network` by default). The files must be named `<person-name>_xxxxxxxx.npy`. If the files were recorded using the provided recording script, they should already have this format. Anything after the first underscore is ignored.

> [!warning] 
> The training script currently acquires its classes from the unique names associated with the files so that multiple samples can be provided per person (e.g. `person1_standing`, `person1_walking`). Therefore, it is probably a good idea to use more than just first names like we do in our example.

# Client

# Server