#!/usr/bin/env bash

STEP=$1

if [[ ${STEP} == "1" ]]; then
    sudo yum update -y
    sudo yum erase nvidia cuda
    sudo yum install -y amazon-efs-utils

    sudo mkdir -p /mnt/efs
    sudo chown ${USER} /etc/fstab
    sudo echo "fs-aeea2a5f:/ /mnt/efs efs defaults,_netdev 0 0" >> /etc/fstab
    sudo chown root /etc/fstab
    sudo mount -a
    sudo chown -R ${USER} /mnt/efs

    cd ${HOME}
    sudo chown -R ${USER}:${USER} /mnt/efs
    wget http://us.download.nvidia.com/tesla/418.67/nvidia-diag-driver-local-repo-rhel7-418.67-1.0-1.x86_64.rpm -O nvidia.rpm
    wget http://us.download.nvidia.com/tesla/418.67/NVIDIA-Linux-x86_64-418.67.run
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b
    ${HOME}/miniconda3/bin/conda init

    mkdir canon
    cp -r /mnt/efs/canon/scripts canon/
    ln -s /mnt/efs/canon/canon canon/canon
    ln -s /mnt/efs/logs canon/scripts/logs
    ln -s /mnt/efs/checkpoints canon/scripts/checkpoints
    pushd ~/canon/scripts/img && unzip img.zip && popd

    # reboot
fi

if [[ ${STEP} == "2" ]]; then
    sudo yum install -y gcc kernel-devel-$(uname -r)

    sudo touch /etc/modprobe.d/blacklist.conf
    sudo chown ${USER} /etc/modprobe.d/blacklist.conf
    echo "blacklist vga16fb" >> /etc/modprobe.d/blacklist.conf
    echo "blacklist nouveau" >> /etc/modprobe.d/blacklist.conf
    echo "blacklist rivafb" >> /etc/modprobe.d/blacklist.conf
    echo "blacklist nvidiafb" >> /etc/modprobe.d/blacklist.conf
    echo "blacklist rivatv" >> /etc/modprobe.d/blacklist.conf
    sudo chown root /etc/modprobe.d/blacklist.conf

    sudo chown ${USER} /etc/default/grub
    sudo echo "GRUB_CMDLINE_LINUX=\"rdblacklist=nouveau\""  >> /etc/default/grub
    sudo chown root /etc/default/grub
    sudo grub2-mkconfig -o /boot/grub2/grub.cfg

    sudo /bin/sh ${HOME}/NVIDIA-Linux-x86_64*.run

    # reboot
fi

if [[ ${STEP} == "3" ]]; then
    sudo nvidia-persistenced
    sudo nvidia-smi --auto-boost-default=0
    sudo nvidia-smi -ac 877,1530

    conda env create -f /mnt/efs/conda-gpu.yaml
fi
