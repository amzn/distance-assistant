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
