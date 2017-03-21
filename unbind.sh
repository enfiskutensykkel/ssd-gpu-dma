#!/bin/bash

if [[ -z "$2" ]]; then
	"Usage: $0 bind|unbind <bus>:<device>.<fn>"
	exit 1
fi

DEVICE=`lspci -s "$2" | grep -oP "\d{2}:\d{2}\.\d+"`

# TODO: namespace and cuda device

if [[ -z "$DEVICE" ]]; then
	echo "Not a valid PCI device: $2"
	exit 1
fi

case "$1" in
	"bind")
		echo 0 > "/sys/bus/pci/devices/0000:$DEVICE/enable"
		echo -n "0000:$DEVICE" > /sys/bus/pci/drivers/nvme/bind
		;;

	"unbind")
		echo -n "0000:$DEVICE" > "/sys/bus/pci/devices/0000:$DEVICE/driver/unbind"
        	echo 1 > "/sys/bus/pci/devices/0000:$DEVICE/enable"
		;;

	"run")
		rmmod cunvme.ko || true
		insmod cunvme.ko num_user_pages=128
		./cuda-nvme 0000:$DEVICE || true
		rmmod cunvme.ko
		;;

	*)
		echo "Usage: $0 bind|unbind <bus>:<device>.<fn>"
		exit 1
		;;
esac
