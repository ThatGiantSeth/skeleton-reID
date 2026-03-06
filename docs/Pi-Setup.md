> [!note]
> As stated in initial setup, this guide assumes that you are using a Windows machine for the client and a Raspberry Pi for the server. These instructions will be slightly different on different operating systems.

# System setup
Start by setting up your Pi. We recommend a Raspberry Pi 4 or newer, but it must be at minimum a Pi 3. We used a standard install of Raspberry Pi OS 5, which is based on Debian 12 "Bookworm". You can follow [this guide](https://www.raspberrypi.com/documentation/computers/getting-started.html) if you do not know how to install an OS to your Pi, so I am only going to discuss the customizations below.

When you use Raspberry Pi Imager to create an SD card, it will present you with several customization options (as mentioned in the guide). There are some important points that I want to highlight about each customization section:
- Hostname: Choose a short and memorable hostname ("friendly name"), as this is what you will use for the server address.
- Username/password: Choose a secure but memorable username and password. You will use these to log into the Pi remotely after we finish setup.
- Wi-Fi: Although we will be using a local Ethernet connection for our client-server connection, a regular Internet connection is required to install dependencies.
- Remote access: This one is very important. We want to be able to run our Pi without a separate monitor and keyboard ("headless"). Ensure that you enable SSH on this page.

After you finish creating your SD card and installing it in the Pi, you can connect a monitor, mouse, and keyboard, Ethernet to the Windows machine, and power. Since we performed customization during the imaging process, it should skip the setup wizard and go straight to the desktop. If it runs the setup wizard anyway, just select the same options as you did in the imager.

There are a few more customizations we will want to perform before we make our Pi "headless":

Since we won't need a desktop environment after this initial setup, we are going to disable booting to the desktop GUI. To do this, navigate to `Application Menu (Raspberry Icon) > Preferences > Raspberry Pi Configuration`. Under the `System` tab, switch the `Boot:` option from `to Desktop` to `to CLI`. In addition, if any of the `Auto Login` toggles are enabled, toggle them off for security. You'll need your password for remote access either way.

The next step is to set up a **static IP address**. When we tried to set this up originally, we found that both SSH and our application had issues with the Pi changing its local IP address. Setting a static IP address for the Ethernet connection solves this problem. In order to do this, we will need to make use of the Pi's `nmtui` CLI interface. To access it, first open the Terminal (the icon is >_ in a black box) and type:
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

# Setting up the Python virtual environment

## [Put next Section Link Here](#)