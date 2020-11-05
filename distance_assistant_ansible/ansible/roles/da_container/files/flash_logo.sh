#!/bin/bash

xloadimage -display :0 /usr/local/etc/social-distancing-signs-24.jpg &
pid=$!
sleep 30
kill $pid
killall xloadimage

