

> [!note] 
> This guide was designed with a Windows client and a Raspberry Pi-based server in mind. These steps will vary on different operating systems.
# Requirements
> [!note] Python package requirements for each component will be discussed in their relevant sections

### Before installing Python, see [Python Install Manager](#python-install-manager-pim) below.

## Training
> [!tip] CUDA-enabled GPU is strongly recommended for training but not required
- ARM or x86-based machine
- NVIDIA CUDA Toolkit (if using CUDA)
- Any PyTorch-compatible version of Python

> [!important]
> See [PyTorch downloads](https://pytorch.org/get-started/locally/) for currently supported CUDA and Python versions. It is important to install a version of CUDA that is supported by PyTorch. Otherwise, torch will use the CPU even if CUDA is installed.

## Client/Recording
- x86-based machine
- OpenNI-compatible camera (Examples: Kinect 360, PrimeSense Carmine 1.08/1.09)
- Windows 10 or 11
- Python 3.6
- [CMake v3.4.3](https://github.com/Kitware/CMake/releases/download/v3.4.3/cmake-3.4.3-win32-x86.exe)
- [OpenNi 2 from structure.io](https://web.archive.org/web/20250912105130/https://s3.amazonaws.com/com.occipital.openni/OpenNI-Windows-x64-2.2.0.33.zip)
- [PrimeSense NiTE 2.2](https://web.archive.org/web/20260305002027/https://bitbucket.org/kaorun55/openni-2.2/raw/2f54272802bfd24ca32f03327fbabaf85ac4a5c4/NITE%202.2%20%CE%B1/NiTE-Windows-x64-2.2.zip)
- Visual Studio 2022 (see [Visual Studio Requirements](#visual-studio-requirements) below)
> [!warning] Do not use the version of OpenNI provided by the BitBucket repository hosting NiTE 2.2. It will not run properly. Use the version from structure.io linked above.

## Server
- ARM or x86-based machine (we used Raspberry Pi)
- Any PyTorch-compatible version of Python

## Python Install Manager (PIM)
> [!note] This section is only required on the Windows machine. We will assume for this guide that the Pi is dedicated to this purpose and only needs one Python installation.

This guide assumes that you do not have a version of Python already installed. However, even if you do have it installed, Python Install Manager is recommended to keep track of multiple versions.

### Python Install Manager can be downloaded from the top of the Python downloads page: https://www.python.org/downloads/

The installer may ask a series of questions depending on your current Python installation:
- Do you want to edit app execution aliases? This option will open Settings > App execution aliases. If you previously installed Python, Windows will have set that installation as the default executable for the commands `python` and `python3`. PIM recommends that you switch this to its own `Python(default)` options, which will allow PIM to manage the default installation instead of Windows.
- If you had the legacy "Python Launcher" application installed, the PIM installer will warn you. PIM is a replacement for Python Launcher so it should be safe to uninstall it in most cases. This will prevent conflicts with the `py` command.
- Do you want to add Python shortcuts to your PATH? This option adds version-specific shortcuts e.g. `python3.11.exe`, `python3.6.exe` to your system's environment variables. **This is required to use these commands directly in Windows CMD/PowerShell.** Otherwise, you will have to reference the absolute path to the Python executable e.g. `C:\Users\<user>\AppData\Local\Python\pythoncore-3.11`.
- Do you want to install the latest Python runtime? If you already have a Python installation, you can skip this. We will be using PIM to install specific versions later anyway.

See the [Python install docs](https://docs.python.org/3/using/windows.html#python-install-manager) for more information.

After the PIM installer completes, you will have to restart your CMD/PowerShell window to use it. After reopening the CLI, you can type `py install <version>` to install a specific version. After installing, it will tell you how to reference that version e.g. `python3.6.exe` to access Python 3.6 instead of the default `python` command.

### As stated above, for this project, you will need to install two versions:
- A PyTorch supported version, e.g. `py install 3.11`
- Python 3.6, installed with `py install 3.6`

## Visual Studio Requirements
The setup scripts for some of the client dependencies make use of some Visual Studio build tools. Therefore, Visual Studio 2017 or later must be installed before setting up the client. **If you do not install VS, you will experience errors when trying to install some packages through `pip`!**

Download the Visual Studio Community Installer from [Microsoft's website](https://visualstudio.microsoft.com/vs/community/). Although this requires a Microsoft account, the Community edition **does NOT** require a subscription! During installation, make sure to check the `Desktop Development with C++` workload. This will install the necessary build tools.

# Setup
First we will set up the virtual environments, the training script, client and recording scripts (they share their dependencies so will use the same virtual environment), and finally the server.

## Creating Python virtual environments
> [!important] 
> Because this project requires specific Python versions for different components, it is **highly recommended** to use virtual environments to compartmentalize your client/server/training packages and versions. We will be using example environment names throughout this guide that match their associated script e.g. `client-env`, `server-env`, and `training-env`. If you do not want to use virtual environments you can skip this section, but the rest of this guide will be more confusing.

## Training Script
The training script `network/train.py` takes in a set of NumPy arrays containing skeleton data, normalizes them, and trains a CNN `network/CNN.py` on this data. It outputs a file `skeleton_model_best.pth`.
### Input data
The CNN is set up to accept skeleton data consisting of 3D world coordinates from an OpenNI-compatible camera like PrimeSense Carmine 1.09 or Microsoft Kinect 360. Any OpenNI camera will work, but the same camera should be used for both training and demonstration for coordinate system consistency. 
The script searches for `.npy` files in a directory called `data`, located within the script's directory (`./network` by default). The files must be named `<person-name>_xxxxxxxx.npy`. If the files were recorded using the provided recording script, they should already have this format. Anything after the first underscore is ignored.

> [!warning] 
> The training script currently acquires its classes from the unique names associated with the files so that multiple samples can be provided per person (e.g. `person1_standing`, `person1_walking`). Therefore, it is probably a good idea to use more than just first names like we do in our example.

### Setting up the environment
The training script requires any of the Python versions supported by PyTorch. As of this documentation, these are **Python 3.11 and newer**. We used Python 3.11 during our project.

## Client and recording script

## Server

remember to explain install of opencv including VS 2022 and Cmake 3.4.3
