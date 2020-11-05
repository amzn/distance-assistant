#!/usr/bin/python3

import subprocess


p = subprocess.Popen(["/usr/bin/v4l2-ctl", "--list-devices"],
                     stdout=subprocess.PIPE)
output, error = p.communicate()

cameras = []
for line in output.decode().split():
    l = line.strip()
    if l.startswith("/dev/video"):
        cameras.append("--device=%s" % l)


print(" ".join(cameras))

