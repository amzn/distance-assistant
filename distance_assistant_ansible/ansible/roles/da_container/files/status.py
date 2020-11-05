#!/usr/bin/python3
import os
import subprocess
import re
from dateutil.parser import parse
from datetime import timedelta, datetime
import json


usb_id = '8086:'


class Check(object):
    def __init__(self, ok, short_error):
        self.ok = ok
        self.short_error = short_error
        self.status = None

    def run(self):
        pass


# does the os detect an nvidia card?
def os_nvidia():
    status = os.system('lsmod | grep -q nvidia')
    if status == 0:
        return 'Nvidia card detected.', None
    return None, 'Nvidia card not found!'


# does the OS detect a realsense camera?
def os_realsense():
    status = os.system('lsusb -d %s -v > /dev/null' % usb_id)
    if status == 0:
        return 'RealSense camera connected.', None
    return None, 'RealSense camera not found.'


# is it running at the proper speed?
def realsense_speed():
    lsusb = subprocess.run(['lsusb', '-v', '-d', '8086:'], stdout=subprocess.PIPE)
    rgx = re.compile('bcdUSB[ ]*([^ ]*)')
    match = rgx.search(lsusb.stdout.decode('utf-8'))
    speed = match.group(1)

    if speed == '3.20\n':
        return 'Realsense connected at full speed.', None
    if speed == '2.10\n':
        return None, 'Insufficient USB connection for RealSense camera.'
    if not speed:
        return None, None
    return None, 'Unknown connection speed for RealSense camera.'


# is the OS online?
def os_online():
    status = os.system('connection_status.sh > /dev/null')
    if status == 0:
        return 'Able to resolve amazon.com.', None
    return None, 'Unable to resolve amazon.com.'


# has the OS provisioning completed?
def provisioned():
    if os.path.exists(ansible_prov_file):
        return 'Initial OS provisioning completed.', None
    return None, 'Initial provisioning incomplete.'


# has os updates completed?
def updated():
    with open(ansible_status_file) as status:
        if 'failed' in status.read():
            return None, 'Update failed.'
    return 'Updates have not failed.', None


# is the service running?
def service():
    status = os.system('systemctl is-active --quiet da')
    if status == 0:
        return 'da service is running.', None
    return None, 'da service is not running.'


# is the docker image downloaded?
def downloaded():
    status = os.system('systemctl status da | grep -q "docker pull"')
    if status == 0:
        return None, 'Docker container is downloading.'
    return 'Docker container is not being downloaded.', None


# is the container running?
def app_running():
    status = os.system('docker ps -q -f name=da | grep -q .')
    if status == 0:
        return 'Docker container is running.', None
    return None, 'Docker container not running.'


# does the container detect nvidia?
def app_nvidia():
    return None, None


time_threshold = -20


# does the container detect a realsense camera?
def app_realsense():
    log = log_highlights.get('no_camera', None)
    if log:
        datestring = log.split(' ', 1)[0]
        time = parse(datestring)
        if time < datetime.utcnow() + timedelta(seconds=time_threshold):
            return 'Camera not detected.', None
    return None, None


# is the camera throwing errors?
def realsense_errors():
    return None, None


log_file = '/var/log/da/console.log'
max_reads = 1000


def reverse_readline(filename, buf_size=8192):
    """A generator that returns the lines of a file in reverse order"""
    with open(filename) as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size))
            remaining_size -= buf_size
            lines = buffer.split('\n')
            # The first line of the buffer is probably not a complete line so
            # we'll save it and append it to the last line of the next buffer
            # we read
            if segment is not None:
                # If the previous chunk starts right from the beginning of line
                # do not concat the segment to the last line of new chunk.
                # Instead, yield the segment first
                if buffer[-1] != '\n':
                    lines[-1] += segment
                else:
                    yield segment
            segment = lines[0]
            for index in range(len(lines) - 1, 0, -1):
                if lines[index]:
                    yield lines[index]
        # Don't yield None if the file was empty
        if segment is not None:
            yield segment


def parse_log():
    output = {}
    key_lines = {
        'meters': 'Initialized camera height',
        'blocking': 'there were not enough ground depth points to compute height',
        'no_camera': 'exactly one camera is expected, but found: 0',
    }
    log = reverse_readline(log_file)
    for i in range(max_reads):
        line = log.__next__()
        if not key_lines:
            break
        for error, string in key_lines.items():
            if string not in line:
                continue
            output[error] = line
            key_lines.pop(error)
    return output


log_highlights = parse_log()


checks = [
    os_nvidia,
    os_realsense,
    realsense_speed,
    os_online,
    provisioned,
    updated,
    service,
    downloaded,
    app_running,
    app_nvidia,
    app_realsense,
    realsense_errors,
]

successes = []
failures = []
for check in checks:
    success, fail = check()
    if success:
        successes.append(success)
    if fail:
        failures.append(failures)

out = json.dumps({
    'ok': successes,
    'fail': failures
})

print(out)
