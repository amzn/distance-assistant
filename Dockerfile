FROM nvidia/cudagl:10.1-devel-ubuntu18.04
LABEL maintainer="distance-assistant@amazon.com"

# Force non interactive for apt installs
ENV DEBIAN_FRONTEND noninteractive

# Set terminal type so colors looks correct.
ENV TERM xterm-256color

# Install cudnn 7
RUN apt-get update \
  && apt-get install -y libcudnn7-dev \
  && rm -rf /var/lib/apt/lists/*

LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Install ROS melodic
# http://wiki.ros.org/melodic/Installation/Ubuntu
RUN echo "deb http://mirror.umd.edu/packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros-latest.list \
  && apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
  && apt update \
  && apt install -y --allow-unauthenticated \
    ros-melodic-ros-base \
    python-rosdep \
  && rm -rf /var/lib/apt/lists/*

# ROS post-installation
RUN rosdep init && rosdep update
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

# Install and update pip
RUN apt update \
  && apt install -y python-pip \
  && pip install -U pip setuptools \
  && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN apt update \
  && apt install -y \
    libatlas-base-dev \
    gfortran \
  && rm -rf /var/lib/apt/lists/*

# Install opencv
RUN apt update \
  && apt-get install -y \
    libopencv-dev \
    libopencv-calib3d-dev \
  && rm -rf /var/lib/apt/lists/*

# install cv2 for python
RUN pip install \
        filterpy \
        future \
        numpy==1.16.6 \
        opencv-python==3.1.0.0 \
        opencv-contrib-python \
        scikit-image==0.10.1 \
        scipy \
        sklearn

# Install realsense ROS wrappers
RUN apt update \
  && apt-get -y --allow-unauthenticated install \
    libcanberra-gtk3-module \
    ros-melodic-image-view \
    ros-melodic-realsense2-camera \
    git \
  && rm -rf /var/lib/apt/lists/*

# Setup catkin workspace
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
RUN mkdir -p /home/catkin_ws/src
WORKDIR /home/catkin_ws/src
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_init_workspace /home/catkin_ws/src'

# Get darknet code
RUN git clone https://github.com/AlexeyAB/darknet.git darknet --depth 1

# Copy Makefile to darknet
COPY Makefile /home/catkin_ws/src/darknet

# Run make in darknet to get shared object filegs
RUN cd /home/catkin_ws/src/darknet && \
    make -j && \
    cp libdarknet.so /home/catkin_ws/src && \
    rm -rf /home/catkin_ws/src/darknet

# Copy and build distance assistance ROS package
COPY distance_assistant /home/catkin_ws/src/DistanceAssistant
ENV PYTHONPATH /home/catkin_ws/src/DistanceAssistant/src:$PYTHONPATH
RUN cd /home/catkin_ws/ && \
    /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_make'

# Clean up left over apt stuff
RUN apt clean \
  && apt autoremove \
  && rm -rf /var/lib/apt/lists/*

# Set up container entrypoint.
# NOTE: The whole point is to keep this container "zero-configuration" for the ops team
# Everything the distance assistance app needs to run should be set in the run.bash file
COPY run.bash /home/catkin_ws/run.bash
ENTRYPOINT [ "/home/catkin_ws/run.bash" ]