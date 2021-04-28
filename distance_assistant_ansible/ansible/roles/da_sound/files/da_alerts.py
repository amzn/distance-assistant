#!/usr/bin/python3
import subprocess
import os
import time


ALERT_MP3 = "/usr/share/sounds/gnome/default/alerts/glass.ogg"
ALERT_REQUEST_FILE = "/tmp/sounds/alerts"


def play_sound():
    mp3_player = subprocess.Popen(["sudo", "-H", "-u", "da", "--login",
                                   "/usr/bin/play", "-q", "-v", "10", ALERT_MP3])
    mp3_player.communicate()


while True:
    try:
        if os.path.exists(ALERT_REQUEST_FILE):
            play_sound()
            os.unlink(ALERT_REQUEST_FILE)

    except Exception:
        pass

    time.sleep(0.1)

