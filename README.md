# Orange Pi Lite Object Detector 

This is a little project I've been working on to get some hands on experience with some Orange Pi SBC devices I have, and Tensorflow. It can run 3-4 FPS object detection using ssd_mobilenet_v2 (confirm! Can't even remember where the original detect.tflite file came from). It keeps a count of all object types that have been encountered and the degree of confidence, and makes these available as a standard Prometheus `/metrics` endpoint that can be scraped for easy visualization in Grafana.


## Hardware

Total cost of the hardware was a [very reasonable ~20USD](https://www.aliexpress.com/item/Orange-Pi-Lite-SET9-Orange-Pi-Lite-512MB-and-2Million-Pixel-Camera-with-wide-angle-lens/32663940765.html), though depending on where you are shipping may affect this. 

* Orange Pi Lite  (12 USD)  
    * Allwinner H3 quad-core @ 1.2GHz
    * 512MB RAM
    * 5V 2A DC power
    * Wifi, Bluetooth and lots more

* 2MP GC2035 camera (4 USD)


## OS Setup and Configuration

### Linux

In order to support the GC2035 camera, I had to go with an old version of Armbian with a 3.4 kernel available on the Armbian archive page for the Pi Lite, eg. `Armbian_5.90_Orangepilite_Ubuntu_xenial_default_3.4.113_desktop.7z`

https://archive.armbian.com/orangepilite/archive/


### Drivers
It's recommended to use the modified GC2035 driver here for better quality video:

https://github.com/avafinger/gc2035

You'll also need to compile the following GStreamer 1.0 plugin to get H.264 video to work:

https://github.com/gtalusan/gst-plugin-cedar


If everything installed according to plan, the camera should be ready to go after running the following:

```bash
sudo sunxi-pio -m "PG11<1><0><1><1>"
sleep 3
sudo modprobe gc2035 hres=1 
sleep 3
sudo modprobe vfe_v4l2
```

```bash
$ dmesg | tail

```

## Software

### Video Pipeline

One of the first big challenges I had with this hardware, was that I could not get OpenCV to work with the video interface. I did, however, have luck getting [motion](https://github.com/Motion-Project/motion) to display frames from the camera at roughly 1-2 FPS. So at least I knew the camera worked. 

The solution ended up being to use GStreamer to get video from the camera via v4l2. GStreamer is fairly low level, but turns out to be quite powerful with its ability to support pipelines.

### Tensorflow Model and Runtime

To make best use of the available CPU resources, the app can be run with a configurable number of workers that run object detection against frames coming from the camera. For best FPS (but also high heat/resource consumption) use four, one per core on the H3 Allwinner chipset. 

### Front-end Application

Nothing special here, just a simple Flask application with a template and a `/metrics` endpoint for Prometheus.

The app makes extensive use of the python `multiprocessing` module to make best use of the quad-core CPU. On start up there is:

* Main flask process
* A process to manage the gstreamer pipeline
* Variable number of tensorflow workers (processes)

### Dashboard

Grafana was used for the dashboard to visualize the data output by the app:


## TODO 

* Gracefully handle/restart on the periodic camera lockups that seem to happen
* Create a script for rebuilding image or at least document packages required for install
* Make code less embarassing