#!/usr/bin/env python
import random
import flask
import gi
import multiprocessing
import io
import detect_tflite
import base64
import binascii
import time
import math
import argparse
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import GObject, Gst, GstApp, GstVideo
import sys
import typing
import queue
import numpy as np
import pickle
from PIL import Image
import cv2
import datetime

# For metrics exposition
from prometheus_client import start_http_server, Summary, Counter, Gauge, make_wsgi_app
from prometheus_client import generate_latest
app = flask.Flask(__name__)

start_time = None
IMG_BUF_LEN = 10
coco_labels = detect_tflite.load_labels('coco_labels.txt')

mgr = multiprocessing.Manager()
img_q = mgr.Queue(maxsize=100)
res_q = mgr.Queue(maxsize=100)
# Shared rolling buffer of images that will be consolidated into the primary
# processed and rendered as a video
shr_tf_img = mgr.list()
# The corresponding list of results computed for the images. Will be provided by
# the worker as a list of results (see detect_objects for structure)
shr_tf_res = mgr.list()
# Null out the post-processing shared buffers
for i in range(0, IMG_BUF_LEN):
    shr_tf_img.append((-1, Image.fromarray(np.zeros((600,800), dtype=np.uint8))))
    shr_tf_res.append({})

shr_args = mgr.Namespace()
gen_fps = mgr.Value('f', 12.0)
# Track two different counters - the number of the frame emitted from the camera, and the seq of the
# image that we put on the queue for processing
cam_seq = mgr.Value('i', 0)
vid_seq = mgr.Value('i', 0)
tf_seq = mgr.Value('i', 0)

# Prometheus metrics
prom_camera_images_captured = Counter('nsbeetle_camera_images_captured', 'Number of camera frames captured')
prom_vid_images_processed = Counter('nsbeetle_vid_images_processed', 'Number of video stills processed')
# TODO Create a histogram or summary that includes allows us to bucket the predictions based on the 
# confidence value from TF. For now just use "low", "medium" and "high"
prom_objects_seen = Counter('nsbeetle_objects_seen', 'Number of objects seen', ['class', 'confidence'])

# Trying to adapt the ideas from:
# http://lifestyletransfer.com/how-to-use-gstreamer-appsink-in-python/

def on_buffer(sink: GstApp.AppSink, data: typing.Any) -> Gst.FlowReturn:
    global start_time
    """Callback on 'new-sample' signal"""
    if not start_time:
        start_time = datetime.datetime.now().timestamp()  
    # Emit 'pull-sample' signal
    # https://lazka.github.io/pgi-docs/GstApp-1.0/classes/AppSink.html#GstApp.AppSink.signals.pull_sample

    # It seems like, even if we don't intend to do anything with the sample, we still have to emit it
    sample = sink.emit("pull-sample")  # Gst.Sample

    elapsed = datetime.datetime.now().timestamp() - start_time
    gen_fps.value = cam_seq.value / elapsed

    if not isinstance(sample, Gst.Sample):
        print("Sample retrieved was corrupted")
        return Gst.FlowReturn.ERROR

    cam_seq.value += 1
    prom_camera_images_captured.inc()
    # FIXME Adjust based on incoming rate
    if cam_seq.value % math.ceil(12 / shr_args.target_fps) != 0:
        return Gst.FlowReturn.OK

    vid_seq.value += 1
    prom_vid_images_processed.inc()
    img_arr = extract_buffer(sample)
    print("Pushing image sample size: {}, incr: {}, q_sz: {}, est. fps: {:.1f}".format(
        img_arr.shape, cam_seq.value, img_q.qsize(), gen_fps.value
    ))
    img_q.put((vid_seq.value, img_arr))
    return Gst.FlowReturn.OK


def extract_buffer(sample: Gst.Sample) -> np.ndarray:
    """Extracts Gst.Buffer from Gst.Sample and converts to np.ndarray"""

    buffer = sample.get_buffer()  # Gst.Buffer
    caps = sample.get_caps()
    caps_format = sample.get_caps().get_structure(0)  # Gst.Structure

    # GstVideo.VideoFormat
    video_format = GstVideo.VideoFormat.from_string(caps_format.get_value('format'))
    w, h = caps_format.get_value('width'), caps_format.get_value('height')
    # c = utils.get_num_channels(video_format)
    buffer_size = buffer.get_size()

    # Change back to 800x900 when YUV decoding in place
    shape = (int(h * 1.5), w) # One channel? 800 * 600 = 720000
    # The YUV420 format will be uint8
    return np.ndarray(shape=shape, buffer=buffer.extract_dup(0, buffer_size), dtype=np.uint8)


def make_pipeline():
    """ Adapted from 
    https://stackoverflow.com/questions/34688897/python-gstreamer-getting-meta-api-for-appsink-buffer"""

    # Need to experiment more with use of queues - these allow GStreamer to internally spin off threads which can 
    # help extra more performance from a device like an Orange Pi which has 4 relatively weak cores, but it may be
    # creating overhead which ultimately doesn't improve performance.
    src =  Gst.ElementFactory.make("v4l2src")
    src.set_property("device", "/dev/video0")
    src.set_property("do-timestamp", 1)
    filt = Gst.ElementFactory.make("capsfilter")
    # filt.set_property("caps", Gst.caps_from_string("video/x-raw,format=NV12,width=640,height=480,framerate=20/1"))
    filt.set_property("caps", Gst.caps_from_string("video/x-raw,format=NV12,width=800,height=600,framerate=12/1"))
    p1 = Gst.ElementFactory.make("cedar_h264enc")
    # p1_q = Gst.ElementFactory.make("queue")
    p2 = Gst.ElementFactory.make("h264parse")
    p3 = Gst.ElementFactory.make("rtph264pay")
    p3.set_property("config-interval", 1)
    p3.set_property("pt", 96)
    p4 = Gst.ElementFactory.make("rtph264depay")
    p5 = Gst.ElementFactory.make("avdec_h264")
    sink = Gst.ElementFactory.make("appsink", "sink")
    pipeline_elements = [src, filt, p1, p2, p3, p4, p5, sink]

    sink.set_property("max-buffers", 10) # prevent the app to consume huge part of memory
    sink.set_property('emit-signals', True) #tell sink to emit signals
    sink.set_property('sync', False) #no sync to make decoding as fast as possible
    sink.connect("new-sample", on_buffer, sink)

    # Create an empty pipeline & add/link elements
    pipeline = Gst.Pipeline.new("test-pipeline")
    for elem in pipeline_elements:
        pipeline.add(elem)
    for i in range(len(pipeline_elements[:-1])):
        if not Gst.Element.link(pipeline_elements[i], pipeline_elements[i+1]):
            raise Exception("Elements {} and {} could not be linked.".format(
                pipeline_elements[i], pipeline_elements[i+1]))
    return pipeline

@app.route("/")
def index():
    ctx = {
        "raw_img_seq": cam_seq.value,
        "tf_seq": tf_seq.value,
        "gen_fps": gen_fps.value

    }
    return flask.render_template('index.html', **ctx)


# New experimental feed based on shared memory buffer
@app.route('/stream')
def stream_shm():
    """Video streaming home page."""
    return flask.render_template('stream.html')


def convert_yuv_to_rgb(img_arr):
    """ These 3 lines of code caused a lot of pain. Could not figure
    out any way of doing the YUV to RGB conversion without using OpenCV, 
    which has convenience functions for this """ 
    rgb = cv2.cvtColor(img_arr, cv2.COLOR_YUV2BGR_I420)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


@app.route("/pic")
def shm_pic():
    s = tf_seq.value - 1
    seq, img = shr_tf_img[s % IMG_BUF_LEN]
    print("Cur tf_seq: {}, seq {} from final image buffer".format(s, seq))
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG', quality=85)
    img_io.seek(0)
    return flask.send_file(img_io, mimetype='image/jpeg', cache_timeout=0)


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error in stream processing: %s: %s\n" % (err, debug))
        print("Trying to restart event loop")
        time.sleep(1)
        loop.run()
        
    return True

################################################
# Streaming functions

def pil_image_to_base64(pil_image):
    buf = io.BytesIO() 
    pil_image.save(buf, format="JPEG")
    b = base64.b64encode(buf.getvalue())
    return binascii.a2b_base64(b)


def frame_gen_shm():
    """Video streaming generator function based on frames in shared memory"""

    def calc_sleep():
        return 1 / shr_args.target_fps

    app.logger.info("Generating frames from shared memory! shr_args: {}".format(shr_args))
    while True:
        seq = (tf_seq.value - 1) % IMG_BUF_LEN
        img_seq, img = shr_tf_img[seq]
        frame = pil_image_to_base64(img)

        s = calc_sleep()
        print("Cur tf_seq {}, calc'd seq {}, got seq from tf {}, sleep time {} ".format(tf_seq.value, seq, img_seq, s))
        time.sleep(s)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return flask.Response(frame_gen_shm(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/metrics')
def mymetrics():
    """ Collect metrics manually rather than try to get DispatcherMiddleware to work """ 
    _update_metric_counters()
    print(prom_objects_seen.collect())
    return flask.Response(generate_latest(), mimetype='text/plain')

######## 

def worker():
    while True:
        seq, img_arr = img_q.get()
        img = convert_yuv_to_rgb(img_arr)
        if not shr_args.skip_od:
            results = detect_tflite.obj_detect_from_pil(img, shr_args.threshold, shr_args.classes)
            detect_tflite.draw_boxes(img, results)
        targ_seq = seq % IMG_BUF_LEN
        print("Processed img seq {} from queue, putting in pos {} , prev tf_seq {}".format(seq, targ_seq, tf_seq.value))
        tf_seq.value = seq
        shr_tf_img[targ_seq] = (seq, img)
        shr_tf_res[targ_seq] = results
        res_q.put(results)

def _update_metric_counters():
    while True:
        try:
            results = res_q.get_nowait()
            prom_camera_images_captured.inc()
            for r in results:
                if r['score'] > 0.70:
                    conf = 'High'
                elif r['score'] > 0.50:
                    conf = 'Medium'
                else:
                    conf = 'Low'
                prom_objects_seen.labels( coco_labels[r['class_id']], conf).inc()
        except queue.Empty: 
            break

def main():
    GObject.threads_init()
    Gst.init(None)

    pipe = make_pipeline()

    # create and event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()

    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # start play back and listed to events
    pipe.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass

    # cleanup
    pipe.set_state(Gst.State.NULL)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run nsbeetle')
    parser.add_argument('--target-fps', type=float, default=1, help='Target FPS')
    parser.add_argument('--skip-od', action='store_true', help='Whether to run object detection')
    parser.add_argument('--classes', nargs='+', type=int, default=None, help='List of object classes to show')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold')
    parser.add_argument('--workers', type=int, default=1, help='Number of TF workers to spawn')

    shr_args = parser.parse_args()
    print("Shared args: {}".format(shr_args))

    with app.app_context():
        print("Attempting to run gstreamer in a separate Process...")
        # Running gstreamer in a separate process, then communicating via Queue with 
        # worker processes that do the Tensorflow object detection heavy lifting
        # This allows much better use of the 4 CPU cores available on the orange pi in addition 
        # to making it easier to terminate the forked process via the daemon flag
        t1 = multiprocessing.Process(target=main, args=(), daemon=True)
        for i in range(0, shr_args.workers):
            w = multiprocessing.Process(target=worker, args=(), daemon=True)
            w.start()
        t1.start()

    app.run(debug=True, use_reloader=False, port=8771, host="0.0.0.0", threaded=False)
