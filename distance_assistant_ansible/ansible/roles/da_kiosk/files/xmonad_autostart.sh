#!/bin/bash

# enable compisiting
xcompmgr &

# hide mouse
unclutter &

# Force all displays to 1080p and set multiple displays to mirroring mode.
# Enumerate active displays into array, sorting to get DVI, DP and eDP ahead of HDMI
declare -a screens
screens=(`xrandr --listactivemonitors | egrep -v '^Monitors' | awk '{ print $4 }' | sort`)

# force 1080p on all displays
for disp in ${screens[@]}; do
	xrandr --output "$disp" --mode 1920x1080
done

# If more than one screen, set each additional to mirror the first.
if [ ${#screens[@]} -gt 1 ]; then
	index=0
	until (( ( index += 1 ) >= ${#screens[@]})); do
		xrandr --output ${screens[$index]} --same-as ${screens[0]}
	done
fi

# set something as background so user knows it booted
hsetroot -add "#000000" -add "#0E2AA8" -gradient 180

# do not turn off screen
xset s 0
xset s off 
xset -dpms

# permit display
xhost +local:root

# display data on desktop
conky &

# allow network config
stalonetray -t --window-layer top -d none &
nm-applet --no-agent &

sleep infinity
