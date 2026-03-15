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

## Test Set
We actually recommend recording **two** separate sets of data for each person, although not strictly required to train the model:
- Your main training set, used to train the model with cross-validation
- A separate test set, used to test your model on a separate dataset. This can be shorter than the training set but should still be longer than 500 frames.

Both of these should be recorded with the exact same environment and script. It could be useful to choose two different output directories for the different sets e.g. `train/` and `test/` using the `-o` option.

When you're done recording your training and test sets, move the training data into a directory called `data` and test data into a directory called `data_test`, both within the `network` folder.

# Training Script
The training script `network/train.py` takes in a set of NumPy arrays containing 3D skeleton data, normalizes them, and trains a CNN `network/CNN.py` on this data. It outputs a folder `fold_models/` containing the best model from each validation fold.

## Input data
The script searches for `.npy` files in a directory called `data`, which should be located within the `./network` directory. The files must be named `<person-name>_xxxxxxxx.npy`. Anything after the first underscore is ignored by the training script. If the files were recorded using the recording script above, they should already have this format.

## Runtime arguments
Like the recording script, the training script has several runtime arguments that allow you to change different parameters:
- `--data-dir`, the directory that the training data is located in
- **!!!** `-w` or `--window_size`, the size of each "window" that the training data is split into, aka how many frames are included in each sample
- `-s` or `--stride`, how much the windows overlap (the closer the stride is to the window size, the less they overlap). This should generally be less than or equal to the window size.
- `-e` or `--epochs`, how many epochs to perform
- `--lr`, the learning rate
- `-k` or `--k_folds`, how many folds to perform for cross validation. also determines the data split between train/validation.
- **!!!** `-d` or `--drop_prob`, random dropout probability
- `-b` or `--batch_size`, number of samples in a single network pass
- `--no_matrix`, disable printing confusion matrices (mainly useful if the dataset is large enough that a confusion matrix would be hard to read)
- `--no_test`, disable validation on a separate test set
  
> [!important]
> There are some arguments, marked with a `!!!`, that also need to be changed on the server running your trained model. These will be mentioned again in the [Server](#server) section.

The script has defaults for each of these arguments (set to the values we used during our training). Therefore, they are not strictly required, but are useful for tuning the model.

## Training the model
If you have all your training data recorded and placed into the `data` directory, you can now run the training script. Assuming you are back in the main project directory:
```sh
cd network
```
```sh
..\network-env\Scripts\python train.py
```

> [!note]
> If you chose not to record a separate test set earlier, use the `--no_test` option to disable test set validation.

This will take quite some time depending on your hardware. As the script runs, it prints several statistics during training like the current fold, current epoch, training accuracy/loss, and validation accuracy/loss. At the end of each fold, it will print a confusion matrix (unless disabled) with the predictions for each class. At the end of the training process, it will validate each fold against a test set (if provided) and print confusion matrices for these as well.

If everything worked properly, you should now have a folder `network/fold_models`, which contains the best model from each of the folds. You can use the test set validation to choose the one with the highest accuracy and/or lowest class bias. The best model is the one that you will transfer to the server. You should also see two other files in the `network/` folder:
- `people_map.json`, which is a class mapping (e.g. class 0 is seth, class 1 is aubrey, etc.)
- `normalization_stats.json`, which saves the minimum and range of your data to ensure consistent normalization between environments

# Server
As with the setup portion of the guide, I will assume you are using a Raspberry Pi for your server. Now that you have completed training, you have some new files that need to be copied over. These files should be copied into the `skeleton-server` directory on your Pi, using WinSCP as you did during setup:
- `normalization_stats.json`
- `people_map.json`
- `fold_x_best.pth`

## Testing the server
First, let's test the server script and set up the arguments. Use PuTTY to connect to the Pi and access the command line as you did during setup. Assuming you set up the folders the same way as the guide, navigate to the server folder:
```sh
cd skeleton_server
```
And run the server with a command like this:
```sh
../server-env/bin/python pi_server.py --model ./fold_x_best.pth --window 10
```
You will need to change `fold_x_best.pth` to match the model file you copied over and set `--window` to the value you set during training. If the server ran successfully you should see an output like this:
```
Loaded normalization stats.
Loaded 4 classes.
Serving on ('0.0.0.0', 5555)
```

## Run script
To make it easier to run the server in the future without remembering all these parameters, there is a provided runner script `run-server.sh`. It just needs to be edited with the correct parameters for your setup. There are two different ways you can edit this script:
- **Using WinSCP**: Simply double click the file and it will open in the default text editor. When you save your changes, WinSCP will automatically update the remote file.
- **Using PuTTY and `nano`:** Connect to the Pi with PuTTY like you did during setup, then use the command `nano run-server.sh` to bring up a CLI text editor. Use your arrow keys to move the cursor and make your edits. To save your file and exit, hit `Ctrl+X`, then `y`, then `Enter`.

Inside `run-server.sh` you will see one long command that may look confusing at first:
```sh
bash -c 'cd ./skeleton_server; screen -d -m -S skeleton_server ../server-env/bin/python pi_server.py --model ./fold_x_best.pth --window 10'
```
Although you won't need to change most of it, it is good to know how it works so you can update it in the future:
- `bash -c` indicates that the command inside the quotes needs to be interpreted as `bash` script. In general, don't change this.
- `cd ./skeleton_server` navigates to the server folder. You should change this if you decided to name your server folder something else.
- `screen -d -m -S skeleton_server` runs the command after it in a background process called a screen. `-d -m` tells it to start *detached* (in the background), and `-S` allows you to set a name for the screen. This name is `skeleton_server` by default, but can be changed to whatever you want. Using `screen` to manage your server is discussed in more detail below.
- Finally, `../server-env/bin/python pi_server.py --model ./fold_x_best.pth --window 10` is the actual Python command that runs the server. It should look pretty similar to the command you ran earlier, and you should edit it with the correct parameters like you did before. You can also add the `--address` and `--port` arguments if you want to change them.

## Using `screen` to manage your server
When you run a Python program on your Pi using PuTTY, it runs inside that terminal environment. Therefore, when you close the connection, the script is interrupted. To work around this, we can use a program called `screen` that allows us to start background tasks.

Screen is an extremely powerful tool, but we will only be using a few of its capabilities for this application:
- Creating a screen: This is shown above with `screen -d -m -S skeleton_server`. You can omit `-d -m` to have the screen start *attached*, meaning you can view its output.
- Attaching to a screen: To attach to a screen, use the `-x` flag with the name you assigned during creation e.g. `screen -x skeleton_server`. This will let you view the screen, type commands into it, and view the output.
- Detaching from a screen: While attached to a screen, press `Ctrl+A` and then release it. This tells `screen` that you are about to type a command. Then press `d` for "detach". This will take you back to the regular CLI.

--- next up: i'm not sure tbh maybe just move on to client section

# Client