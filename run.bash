#!/bin/bash

# strict mode
set -eo pipefail
IFS=$'\n\t'

# This script is execute when the docker container starts.
chmod +x /home/catkin_ws/src/DistanceAssistant/scripts/distance_assistant_node.py
source /home/catkin_ws/devel/setup.sh
roslaunch distance_assistant distance_assistant.launch