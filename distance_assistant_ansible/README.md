# Ansible Playbooks

These are some Ansible playbooks to help assist in setting up Distance Assistant.  They are broken into different roles in case some things need to be done differently.

The instructions below will cover how to manually run these playbooks on the host system.  This will allow uses unfamiliar with Ansible to use them. 

# How to use

If you are unfamiliar with Ansible the section can help. 


Install requirements and pull this code:
```
apt-add-repository --yes --update ppa:ansible/ansible
apt-get -y install ansible
# replace this with public later
git clone ssh://averdow@git.amazon.com/pkg/AmazonDistanceAssistantPublicKiosk
cd AmazonDistanceAssistantPublicKiosk/distanceassistant-ansible/ansible
```

The sections below correspond to individual playbooks that can be run like so:

```
ansible-playbook -i "localhost," all.yml --tags playbook
```

The playbooks below can be installed individually or everything can be run at once with the "everything" playbook.

```
ansible-playbook -i "localhost," all.yml
```

# Base Requirements

These are requirements to run the Distance Assistant application.  With this section complete the container can be manually launched.

## Docker Setup

Configure docker so that it can use the GPU.

```
ansible-playbook -i "localhost," all.yml --tags docker
```

## Nvidia driver Setup

Compile and install nvidia drivers known to work with the Distance Assitance application.

```
ansible-playbook -i "localhost," all.yml --tags nvidia
```

## Intel RealSense drivers

Compile and install RealSense drivers as well as the camera configuration utility. 

```
ansible-playbook -i "localhost," all.yml --tags realsense
```

## Build Container

This will build the docker container to run the application.  This will automatically run the instructions covered by the base README.

```
ansible-playbook -i "localhost," all.yml --tags users
ansible-playbook -i "localhost," all.yml --tags usb-reset
ansible-playbook -i "localhost," all.yml --tags da_container
```

# Kiosk

The playbooks in this section will help launch the application automatically.  With this section complete the computer should boot up and automatically show the distance assistant application.

```
ansible-playbook -i "localhost," all.yml --tags da_kiosk
```

## System Service

This adds a simple systemd service that will automatically launch the application when the computer boots.

## Kiosk Mode

Sets up an unpriviliged user to display the application.  The display manager and window manager will be configured to autmoatically log in and display the application full screen.

```
ansible-playbook -i "localhost," all.yml --tags da-kiosk
```

## udev rule

This adds a udev rule that restarts the application every time a realsense camera is plugged in.  This makes recovery much easier for non-IT users.

# Troubleshooting
