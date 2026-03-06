> [!note]
> As stated in initial setup, this guide assumes that you are using a Windows machine for the client and a Raspberry Pi for the server. These instructions will be slightly different on different operating systems.

# Operating system setup
Start by setting up your Pi. We recommend a Raspberry Pi 4 or newer, but it must be at minimum a Pi 3. We used a standard install of Raspberry Pi OS 5, which is based on Debian 12 "Bookworm". You can follow [this guide](https://www.raspberrypi.com/documentation/computers/getting-started.html) if you do not know how to install an OS to your Pi, so I am only going to discuss the customizations below.

When you use Raspberry Pi Imager to create an SD card, it will present you with several customization options (as mentioned in the guide). There are some important points that I want to highlight about each customization section:
- Hostname: Choose a short and memorable hostname ("friendly name"), as this is what you will use for the server address.
- Username/password: Choose a secure but memorable username and password. You will use these to log into the Pi remotely after we finish setup.
- Wi-Fi: Although we will be using a local Ethernet connection for our client-server connection, a regular Internet connection is required to install dependencies.
- Remote access: This one is very important. We want to be able to run our Pi without a separate monitor and keyboard ("headless"). Ensure that you enable SSH on this page.

After you finish creating your SD card and installing it in the Pi, you can connect a monitor, mouse, and keyboard, Ethernet to the Windows machine, and power. Since we performed customization during the imaging process, it should skip the setup wizard and go straight to the desktop. If it runs the setup wizard anyway, just select the same options as you did in the imager.

There are a few more customizations we will want to perform before we make our Pi "headless":

Since we won't need a desktop environment after this initial setup, we are going to disable booting to the desktop GUI. To do this, navigate to `Application Menu (Raspberry Icon) > Preferences > Raspberry Pi Configuration`. Under the `System` tab, switch the `Boot:` option from `to Desktop` to `to CLI`. In addition, if any of the `Auto Login` toggles are enabled, toggle them off for security. You'll need your password for remote access either way.

## Setting up a static IP address
 When we tried to set this up originally, we found that both SSH and our application had issues with the Pi changing its local IP address. Setting a static IP address for the Ethernet connection solves this problem. In order to do this, we will need to make use of the Pi's `nmtui` CLI interface. To access it, first open the Terminal (the icon is >_ in a black box) and type:
```sh
nmtui
```

You can navigate this interface with your arrow keys and enter key. Navigate to the `Edit a connection` option, then under `Ethernet` pick the wired connection. Change these options:
- Set `IPv4 Configuration` to `Manual` and choose `<Show>` to display the advanced configuration.
- Under `Addresses`, choose `<Add...>`. In the box that appears, type an IPv4 addess of the form `192.168.xxx.xxx` where each `xxx` is a number between 1-255 e.g. `192.168.137.2`. This will be the alternate address you can connect to instead of the hostname you created earlier.
- Navigate down to the option `Require IPv4 addressing for this connection` and enable it.
- Make sure `Automatically Connect` is checked.
- Finally, click `<OK>` to save the settings.

You can exit `nmtui` by selecting `<Back>` on the connection list and then `<Quit>`.

# Setting up remote access
It is more convenient to be able to control the Pi from your main machine, rather than having to plug in a separate monitor, keyboard, and mouse. Therefore, we will use SSH to control to the Pi over the Ethernet connection. However, the disadvantage is that SSH does not forward the desktop GUI, so we will be using the CLI from this point onwards. In addition, we will need a way to transfer files to the Pi.

The first step is to set up an SSH and SFTP client on your Windows machine. I recommend [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/) for SSH and [WinSCP](https://winscp.net/eng/download.php) for SFTP.

## Setting up PuTTY
After installing and running PuTTY, you will see a `PuTTY Configuration` window. Here are the steps to setting up your session:
- If the `Session` tab is not selected by default, select it.
- Host name: Type the host name you created during Pi setup OR the static IP address created above.
- Port: Leave the default value of 22 unless you changed it on the Pi.
- Connection type: SSH.
- Save your session by typing a name into the box below `Saved Sessions` and clicking `Save`.
- Finally, click `Open` to start the SSH session.

At this point, you will probably see a warning about an unknown host key. This is normal for a server that you have never connected to, and you can click `Accept`. If it doesn't connect at all, double check the host name. In addition, you may need to go back to [Setting up a static IP address](#setting-up-a-static-ip-address) to ensure it is set up properly. If all is well, you should be prompted for your username and password that you created previously. After entering your credentials, you should see a terminal prompt like the one you used to configure `nmtui`.

## Setting up WinSCP
After installing and running WinSCP, you should see a `Login` window. If the window does not appear automatically, click the `New Tab` button. In the Login window, follow these steps:
- Click `New Site`.
- Leave `File Protocol` as "SFTP".
- Host name/Port: Use the same host name and port you used above.
- User Name/Password: One difference from PuTTY is that you enter the username and password *before* starting the connection. Enter the same username and password you used above.
- Before clicking `Login`, click `Save` and give the profile a name. You can also choose to save the password to avoid entering it again.
- Finally, click `Login`. You may encounter another "unknown host key" warning, which you can accept as before.

On the left, you should see the files on your main machine and on the right, the files on the Pi. You now have remote access to the files on the Pi.

# Setting up the Python virtual environment
The first thing we will do with our remote access is set up a Python virtual environment just like in the first part of the guide. Some of the syntax will be different since we are working with a Linux-based system instead of Windows, but the process itself is the same. First, check which version of Python was installed by default with Raspberry Pi OS:
```sh
python --version
```

As long as you are using an up-to-date version of Raspberry Pi OS, this version should be compatible with PyTorch, but it is good to make sure.

At this point you can create the virtual environment in the same way as before:
```sh
python -m venv server-env
```

Then, update pip and install PyTorch like you did for the training environment:
```sh
<server-env>/bin/python -m pip install --upgrade pip
```
```sh
<server-env>/bin/pip install torch torchvision
````

## [Running the System](./Running-The-System)