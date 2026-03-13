# Recording Data
A simple script is provided to record skeleton data frames for the training and test sets. It is located in the `client` folder simply because it uses many of the same dependencies as the client.

## Setting up a training environment
We recommend creating a reproducible training prodedure that you can follow before getting started. This will hopefully reduce sources of error during training.

First, find a suitable environment to record in. You should be able to stand far enough from the camera that your whole body fits in frame. We recommend using the `SimpleViewer` app, which is installed with OpenNI, to view the camera feed and find the optimal height/angle for recording.

You will probably want to find an area with minimal background distractions. When you find the optimal subject positioning, mark it with something. We used pieces of masking tape with marker labels. For still recordings, this might just be a single piece of tape that shows where to stand. For a moving recording, this would probably be a marker at the "start" and "end" of the walking area.

## Testing your NiTE installation
Set up your camera in your recording environment and connect it to your PC. Open a CMD/Powershell window in the project root folder `skeleton-reID` and navigate to the client folder:
```sh
cd client
```
> [!note]
> The following commands assume that you have set up your virtual environments in the same way as the guide e.g. located within the project root and with the specified names.

From here, you should be able to run the script. Try doing a test run before actually recording any data to make sure that OpenNI/NiTE are installed properly:
```sh
..\client-env\Scripts\python trainingData.py --person none --pose testing --batch_size 0
```

If OpenNI/NiTE are working properly, you should see a prompt to enter the name of the person you are recording. If you see an OpenNI error (ONI.StatusError, etc.), make sure that the correct version of OpenNI was installed properly and that you copied the NiTE files into the `client` folder.

Assuming everything is working correctly, you should see an output like this:
```
Device Name: PS1080
Recording batch of 0 frames for "none"...
Successfully saved 0 frames for none. It took 0.013s.
```
## Runtime arguments
As you can see from the test run, the script has several runtime arguments that are used to control it. If you run the script without any arguments you will see this output:
```
usage: recordData.py [-h] [-w WINDOW_WIDTH] [--no_video] --person PERSON
                     --pose POSE [-o OUT_DIRECTORY] --batch_size BATCH_SIZE
recordData.py: error: the following arguments are required: --person, --pose, --batch_size
```
The arguments in `[]` are optional, while the others are required. The error will also tell you which required options are missing. We can see that the recording script requires the `--person`, `--pose`, and `--batch_size` arguments. 

Here is a list of all arguments for this script and their purpose:
- `-h` shows the usage guide (this is a default python argument)
- `-w` changes the width (in pixels) of the window that displays the camera feed during recording
- `--no_video` is a boolean option that disables saving the actual video frames from the recording if you don't need them
- `--person` is the name of the person you are recording
- `--pose` is the pose that you are recording (e.g. walking or standing)
- `-o` lets you pick which folder to save the output to
- `--batch_size` lets you change how many frames to record

## Recording data
> [!warning] 
> The training script currently determines its classes from the unique names associated with the files so that multiple samples can be provided per person (e.g. `person1_standing`, `person1_walking` would both be assigned class `person1`). Therefore, it is probably a good idea to use more than just first names like we do in our examples (using "John_Doe" instead of just "John").
>
> However, the pose name (standing, walking, etc) has no bearing on the training process. The pose name is just useful for organizing the recording files when you have multiple recordings per person.

To take your first recording, start the script with the following command, setting the arguments as necessary. An example:
```sh
..\client-env\Scripts\python trainingData.py --person seth --pose standing --batch_size 1500
```
When choosing your batch size, remember that the calibration period (500 frames) will be trimmed by the training script, so you should ideally record at least double that. We used 1500 frames in our testing.

As soon as you run the script, the person being recorded should step in front of the camera. As soon as NiTE detects them, a window will appear showing the camera feed. If it doesn't appear (mostly happens with standing recordings), have the person take a step forward and then backwards (it's OK if they need to reposition a bit at the beginning because the first 500 frames are dropped during training). As soon as the window appears, the recording has started.

Have the subject keep up their pose for the length of the recording. When the recording is finished, the window will close and the console will show an output like this:
```
Successfully saved 1500 frames for seth. It took 120.076s.
```
You should now be able to see the `.npy` recording (and video frames if enabled) in the output folder you selected (`.\output` by default). You can then repeat this process for more people and poses.

# Training Script
The training script `network/train.py` takes in a set of NumPy arrays containing skeleton data, normalizes them, and trains a CNN `network/CNN.py` on this data. It outputs a folder `fold_models/` containing the best model from each validation fold (see [section that doesn't exist yet]).
## Input data
The CNN is set up to accept skeleton data consisting of 3D world coordinates from an OpenNI-compatible camera like PrimeSense Carmine 1.09 or Microsoft Kinect 360. Any OpenNI camera will work, but the same camera should be used for both training and demonstration for coordinate system consistency. 
The script searches for `.npy` files in a directory called `data`, located within the script's directory (`./network` by default). The files must be named `<person-name>_xxxxxxxx.npy`. If the files were recorded using the provided recording script, they should already have this format. Anything after the first underscore is ignored.


--- add a section about tuning the model (parameters, etc.) make sure to include that things like drop_prob need to be changed in both the training and server script

# Client

# Server