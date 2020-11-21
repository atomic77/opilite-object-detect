#!/bin/bash

set -e

sudo sunxi-pio -m "PG11<1><0><1><1>"
sleep 3
sudo modprobe gc2035 hres=1
#sudo modprobe gc2035 hres=0 mclk=34
sleep 3
sudo modprobe vfe_v4l2
