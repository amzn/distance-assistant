# Compiling for Ubuntu 20.04

Clone the librealsense repository
```bash
git clone https://github.com/IntelRealSense/librealsense.git && cd librealsense
```

Copy the scripts and patches in this directory to the `scripts` directory:
```
cp $PATH_TO_DISTANCE_ASSISTANT/librealsense/*.sh ./scripts/
cp $PATH_TO_DISTANCE_ASSISTANT/librealsense/*.patch ./scripts/
```

Install dependencies
```
sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
```

```bash
mkdir build && cd build
```

Build the Makefile.
```bash
cmake ../ -DFORCE_RSUSB_BACKEND=true -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=true -DBUILD_PYTHON_BINDINGS:bool=true
```

Install the drivers
```bash
make && sudo make install && cd ..
```

**Only if Secure Boot is enabled**: Disable Secure Boot by using the `mokutil` tool:

```bash
# Install mokutil
sudo apt install mokutil

# Disable Secure Boot
sudo mokutil --disable-validation
```

You will need to reboot your computer after running the `mokutil` tool.

Finally, run the installation script
```bash
./scripts/patch-realsense-ubuntu-focal-lts.sh
```

You should now be able to use the `realsense-viewer` tool to use the camera.

# Troubleshooting

## When compiling, I get an error about a missing method in a GLUT library.

You can try installing the freeglut library, and running both CMake and make again:

```bash
# Install freeglut
sudo apt install freeglut3-dev

# Run CMake
cmake ../ -DFORCE_RSUSB_BACKEND=true -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=true -DBUILD_PYTHON_BINDINGS:bool=true

# Run make
make && sudo make install
```
