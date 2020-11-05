#!/bin/bash
# Detects if attached USB device w/ given vendor ID is connected at USB 3.1, 
# USB 2.0 (not fast enough in our case), or not at all

set -euo pipefail

# Set USB Device Vendor ID - currently for Intel RealSense 435i but can be 
# overridden with $1
deviceID=${1:-8086} 

# bcdUSB contains the USB Speed class for the deivce. lsusb always dumps 
# something to STDERR so we're /dev/null-ing it. empty response means the 
# device wasn't found, so we set DISC instead.
USBBusSpeed=$(lsusb -d ${deviceID}: -v  2>/dev/null | grep bcdUSB | awk '{ print $2 }' || echo "DISC")

# If for some reason the command returns nothing, set to disconnected
if [ -z "$USBBusSpeed" ]; then
	USBBusSpeed="DISC"
fi

case "$USBBusSpeed" in
"3.20" ) 
#	echo "USB 3.1 - Good"
	exit 0
;;
"2.10")
	echo '${hr}'
	echo '${color red}Error:${color yellow} Camera connected at USB 2.0 - Check Cable${color}'
	exit 1
;;
"DISC")
	echo '${hr}'
	echo '${color red}Error: Camera not found! Check Cable${color}'
	exit 2
;;
# Really shouldn't get here. Echo to STDERR if we do to avoid showing in conky
* )
	>&2 echo "Camera detect script error"
	exit 255
;;
esac
