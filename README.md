# DistanceAssistant

## Introduction

DistanceAssistant is a social distancing monitor which visually alerts people when they are not observing proper social distancing guidelines. The application is packaged and deployed as a docker container.

Inside of the Docker container, this application uses the Robot Operating System (ROS) as the underlying framework for runtime, launch, and configuration. For more details, see: https://www.ros.org/

## Host Setup

This project has been run/tested on a machine with the following specifications/software installed:

* Ubuntu 18.04
    * Other operating systems may also work, but exposing the RealSense camera USB to VMs is incredibly tricky. Thus, a Linux OS with native Docker support is recomended.
* An Nvidia GPU
    * The GPU must support CUDA as mentioned below
    * We recommend a GTX 1070 or better
* CUDA 10.0 or greater: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
* An [Intel DS435i Realsense camera](https://www.intelrealsense.com/depth-camera-d435i/)
    * Other models may work, but this is the only model that was tested
* [docker-ce](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)
    * Tested on Docker 19.03, but older/newer versions may work
* [nvidia container toolkit](https://github.com/NVIDIA/nvidia-docker)
* Intel RealSense SDK
    * Intel realsense drivers can be installed by following the directions here: https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md.
    * **Note**: the pre-built dkms drivers provided by Intel did not work for us. We compile the drivers using the instructions in the librealsense README for compilation.

**NOTE: It is recommended that the RealSense camera be attached to the PC via a USB 3.1 port with a powered USB hub or an "active" (powered) USB cable. During testing on various PCs, we have seen intermittent issues due to lack of power on the USB port. This can happen across two PCs with the same model number. A powered USB3 hub greatly improves the reliability of the RealSense camera**

### Example Host Setup

To follow along with an example host setup from a fresh installation of Ubuntu 18.04, see the [Example Host Setup Instructions](https://github.com/amzn/distance-assistant/wiki/Example-Host-Setup-Instructions).

### Docker configuration

The following instructions use the ``--gpus=all`` flag to enable the NVIDIA GPU inside of the Docker container. However, for older versions of Docker (pre-19.03), the flag would be ``--runtime=nvidia``. See the NVIDIA Container Toolkit instructions above for more.

**Ubuntu 18.04 Alternative Method**

An alternative method to passing the ``--gpus`` flag would be to modify the ``/etc/docker/daemon.json`` file to use NVIDIA as the default runtime.

To do so, install the nvidia-container-runtime:

```
sudo apt install nvidia-container-runtime
```

Then, add the following to your ``/etc/docker/daemon.json``:

```
"default-runtime": "nvidia",
"runtimes": {
    "nvidia": {
        "path": "/usr/bin/nvidia-container-runtime",
        "runtimeArgs": []
    }
}
```

Finally, restart Docker:

```
sudo service docker restart
```

Now, you can omit the ``--gpus`` flag from the runtime Docker command.

## Build Instructions

To build the docker image, first download the yolo model weights and build the docker file. The weights should be stored in the ``distance_assistant/src`` folder.

```bash
# cd <distance-assistant folder>
# The below command assumes that it's run in the root folder of the
# repository
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights  -P distance_assistant/src/
docker build --network=host . -t distance_assistant/prototype

# The source tree should look like this:
# <root>/
#    distance_assistant/
#      scripts/
#      msg/
#      src/
#        yolov4.weights
#    Dockerfile
#    README.md
```

## Camera Mount Instructions
We suggest to mount the camera between 1.7 and 2.1 meters above the ground and slightly tilted down (~10 degrees) in order to maximize the usable depth sensor FOV.

During the application startup we perform an automated calibration step which computes the camera pitch & roll as well as it's height from the ground plane. Therefore it's important that the camera is stationary and there are no large objects in front of the camera blocking the ground when you launch the application.

## Local Execution Instructions

To execute a locally built docker image,

```bash
# IMPORTANT!!
# allow connections to X server
xhost +local:root

# NOTE: The realsense camera should be connected to the host.
docker run \
    --gpus=all \
    --net=host \
    --privileged \
    --device=/dev/usb \
    --device=/dev/media0 \
    --device=/dev/media1 \
    --device=/dev/video0 \
    --device=/dev/video1 \
    --device=/dev/video2 \
    --device=/dev/video3 \
    --device=/dev/video4 \
    --device=/dev/video5 \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
    -it \
    distance_assistant/prototype
```

# Troubleshooting

## Display Failures

You might see errors like this:

```
[ INFO] [1592853388.649425112]: Using transport "raw"
No protocol specified
Unable to init server: Could not connect: Connection refused

(vis_img:105): Gtk-WARNING **: 19:16:28.652: cannot open display: :1

[image_view-5] process has died [pid 105, exit code -11
```

These errors mean that Docker does not have access to the X11 display server. To fix this issue, ensure that you run the following command before starting the Docker container via ``docker run``:

```
xhost +local:root
```

# Distance Assistant Kiosk Setup

To turn the current host into a Distance Assistant Kiosk, one can
take advantage of the included ansible files.

The kiosking process will alter the existing system to be a dedicated
purpose device for running distance assistant, it will boot straight into
a limited purpose window manager and run distance assistant in the background.

It is not recommended to do this with a general use laptop or desktop.

Requirements: Ubuntu 18.04.5 or higher on an x86/64 system matching the above
hardware specifications.

## Install Ubuntu

### Configure BIOS
Go into BIOS, disable secure boot, disable trusted computing

### Download Ubuntu image

Download an Ubuntu installation image, e.g. https://releases.ubuntu.com/18.04/ubuntu-18.04.5-desktop-amd64.iso

Make sure that this is 18.04.5 or later; earlier versions likely won’t work

### Create bootable image from it

Follow this procedure:

https://ubuntu.com/tutorials/create-a-usb-stick-on-ubuntu#1-overview

### Perform the Installation

1. Boot the image
⋅⋅⋅May need to follow prompts on bios to select the USB disk..

2. Select install Ubuntu

3. Select English (US) keyboard layout

4. Select “Minimal installation”

5. Check download updates while installing

6. Check install third-party software for graphics and wifi hardware

7. If you see a checkbox for secure boot, reboot and go back to [Configure BIOS](#configure-bios)

8. Select “Erase Disk and Install Ubuntu” (no LVM/Encryption)

9. Select appropriate timezone

10. Provide User Credentials
    1. Your name: DistanceAssistant
    2. Your Computer’s Name: dakiosk
    3. Pick a username: provision
    4. Choose a password: changeme
    5. Confirm your password: changeme

11. Let system install and restart

### Log In
<a name="Login"></a>

1. log in as provision/changeme
2. Pull up a terminal window (Ctrl+Alt+T); or waffle, search for terminal

### Download, clone, or copy this Repository to the Device

Download or copy the data from this github repository onto the device.

1. Install git:
```
sudo apt-get install git
```

2. Get the repository URL. In the upper right of this github repository, click "code", and copy the download URL

3. Clone the repository:

```
git clone <paste link copied from right pane here>
```

4. Change directory into the repository

```
cd distance-assistant
```



### Install Ansible:

```
sudo apt-add-repository --yes --update ppa:ansible/ansible
sudo apt-get -y install ansible
```

### Edit group_vars

1. Edit the file distance_assistant_ansible/ansible/group_vars/all

1. Change:
    1. da_container_id: set this to the docker tag of the container to use. This defaults to using local image built by the ansible playbooks or by following the [Build Instructions](#build-instructions)

       Hosting your container image on Dockerhub or AWS ECR will avoid having to build the container on each host.

    2. dnsname_for_healthcheck: set this to the dnsname distance assistant should resolve to confirm that it had basic internet connectivity. This defaults to "amazon.com"

### Use Ansible to Downgrade the Kernel

Users have reported better stability with the 5.3.x kernels. The first step
is using ansible to downgrade the kernel.


This must be executed in the top level directory of this package.

```
sudo ansible-playbook -i "localhost," \
    --extra-vars "base_dir=`pwd`" \
    ./distance_assistant_ansible/ansible/all.yml \
    --tags set_53_kernel
```

Once the playbook completes execution, the system will reboot.


### Use Ansible to Run Kiosking Playbooks

After the system comes back up, [log in again](#log-in), cd into the distance-assistant directory and run the rest of the kiosking playbooks.

This must be executed in the top level directory of this package.

```
sudo ansible-playbook -i "localhost," \
    --extra-vars "base_dir=`pwd`" \
    ./distance_assistant_ansible/ansible/all.yml
```

### Finished

When complete, the system should reboot to a blue screen and then start distance assistant. If you configured a remote repository, the station may need to download the image which will take some time.

## Known Issues:

### Black Screen after booting

Some devices have both Intel and NVidia GPUs, try adjusting which one is being used:

```
sudo prime-select intel
```

Then reboot

