#!/bin/bash

###
# This script for the moment hasn't been tested with the armbian custom image build
# process as I haven't had a chance to investigate whether it would be possible against
# such an old version
# 
# This should be run against a fresh install of Ubuntu xenial 5.90 with a 3.4 series kernel
# that supports the GC2035 chipset
#
# The image can be downloaded here:
# https://archive.armbian.com/orangepipc/archive/Armbian_5.90_Orangepipc_Ubuntu_xenial_default_3.4.113_desktop.7z
# or
# https://archive.armbian.com/orangepilite/archive/Armbian_5.90_Orangepilite_Ubuntu_xenial_default_3.4.113_desktop.7z
# on pipc or pilite respectively

sudo apt update && sudo apt upgrade

# Disable gui boot
sudo systemctl set-default multi-user.target

# TODO Looks like I never successfully compiled this, but copied gc2035.ko from a 
# forum post somewhere :-O gc2035.ko copied into folder as a backup, but should
# figure out how to compile this
git clone https://github.com/avafinger/gc2035.git
git clone https://github.com/gtalusan/gst-plugin-cedar
git clone https://github.com/atomic77/opilite-object-detect

sudo pip3 install --upgrade pip

pkg="python3-flask python3-numpy python3-gst-1.0 python3-pil python3-pip "
pkg+="python3-setuptools python3-dev cmake libtool "

sudo apt install ${pkg} -y

# Get tflite runtime. pip3 will need to compile numpy from source. Grab a coffee
wget https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp35-cp35m-linux_armv7l.whl
sudo -H pip3 install tflite_runtime-2.5.0-cp35-cp35m-linux_armv7l.whl

# Alas we need to compile opencv 3.4 from source to be able to use the python3 
# bindings on this old version of Armbian, just for a single color conversion
# function. Sigh. Grab lunch, it takes a while
mkdir ~/src
cd ~/src
wget https://github.com/opencv/opencv/archive/3.4.12.tar.gz
tar xzf 3.4.12.tar.gz

mkdir opencv-3.4.12/build 
cd build

cmake -DCMAKE_INSTALL_PREFIX=/home/atomic/local -DSOFTFP=ON \ 
    -DBUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_opencv_python2=0 \
    -D BUILD_opencv_python3=1 -D WITH_GSTREAMER=ON \ 
    -D PYTHON3_INCLUDE_PATH=/usr/include/python3.5  ..

make -j 4
make install

# Check that ~/local/lib/python3.5/dist-packages has the cv2 shlib
# To get the cv2 python package to load, export to PYTHONPATH
export PYTHONPATH=/home/atomic/local/lib/python3.5/dist-packages


sudo pip3 install prometheus_client

#################################################################
# GStreamer 

# TODO Figure out if we really need all of this
 sudo apt install gir1.2-gst-plugins-base-1.0* gir1.2-gstreamer-1.0* \
    libgstreamer-plugins-base1.0-dev* libgstreamer1.0-dev \ 
    gstreamer1.0-plugins-good

sudo apt install gstreamer1.0-plugins-ugly
sudo apt install gstreamer1.0-plugins-bad

cd ~/gst-plugin-cedar
./autogen.sh
make
# This will copy gst-cedar libraries to /usr/local/lib/gstreamer-1.0
# On pipc, needed to copy this to ~/.local/.. for gstreamer to pick up cedar h264
sudo make install 
cp /usr/local/lib/gstreamer-1.0/libgstcedar.* ~/.local/share/gstreamer-1.0/plugins/

#################################################################
## Grafana and prometheus

wget https://dl.grafana.com/oss/release/grafana_7.3.3_armhf.deb
sudo dpkg -i grafana_7.3.3_armhf.deb
sudo systemctl enable grafana-server
sudo systemctl start grafana-server

wget https://github.com/prometheus/prometheus/releases/download/v2.22.2/prometheus-2.22.2.linux-armv7.tar.gz
tar xzf prometheus-2.22.2.linux-armv7.tar.gz
sudo cp prometheus-2.22.2.linux-armv7/prometheus /usr/local/bin

# Enable service to run on start
sudo cp ~/opilite-object-detect/scripts/prometheus.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable prometheus
sudo systemctl start prometheus
