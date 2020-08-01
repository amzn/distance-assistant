#!python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Both the GPU and no-GPU version should be compiled; the no-GPU version should be renamed "yolo_cpp_dll_nogpu.dll".

On a GPU system, you can force CPU evaluation by any of:

- Set global variable DARKNET_FORCE_CPU to True
- Set environment variable CUDA_VISIBLE_DEVICES to -1
- Set environment variable "FORCE_CPU" to "true"


To use, either run performDetect() after import, or modify the end of this file.

See the docstring of performDetect() for parameters.

Directly viewing or returning bounding-boxed images requires scikit-image to be installed (`pip install scikit-image`)


Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn
@date: 20180503

Portions Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""
#pylint: disable=R, W0401, W0614, W0703
from ctypes import *
import math
import random
import os

PREFIX = os.path.dirname(os.path.abspath(__file__))


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("w", c_float), ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX), ("classes", c_int), ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)), ("objectness", c_float),
                ("sort_class", c_int), ("uc", POINTER(c_float)),
                ("points", c_int), ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int), ("sim", c_float), ("track_id", c_int)]


class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int), ("dets", POINTER(DETECTION))]


class IMAGE(Structure):
    _fields_ = [("w", c_int), ("h", c_int), ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int), ("names", POINTER(c_char_p))]


#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("libdarknet.so", RTLD_GLOBAL)
hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ['PATH'] = cwd + ';' + os.environ['PATH']
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = list()
    for k, v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            else:
                print("Flag value '" + tmp + "' not forcing CPU mode")
        except KeyError:
            # We never set the flag
            if 'CUDA_VISIBLE_DEVICES' in envKeys:
                if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
                    raise ValueError("ForceCPU")
            try:
                global DARKNET_FORCE_CPU
                if DARKNET_FORCE_CPU:
                    raise ValueError("ForceCPU")
            except NameError:
                pass
            # print(os.environ.keys())
            # print("FORCE_CPU flag undefined, proceeding with GPU")
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was
            # compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print(
                "Environment variables indicated a CPU run, but we didn't find `"
                + winNoGPUdll + "`. Trying a GPU run anyway.")
else:
    lib = CDLL("/home/catkin_ws/src/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE, c_char_p]


def network_width(net):
    return lib.network_width(net)


def network_height(net):
    return lib.network_height(net)


predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [
    c_void_p, c_int, c_int, c_float, c_float,
    POINTER(c_int), c_int,
    POINTER(c_int), c_int
]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_batch_detections = lib.free_batch_detections
free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [
    c_void_p, IMAGE, c_int, c_int, c_int, c_float, c_float,
    POINTER(c_int), c_int, c_int
]
network_predict_batch.restype = POINTER(DETNUMPAIR)


def array_to_image(arr):
    import numpy as np
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    """
    Performs the meat of the detection
    """
    #pylint: disable= C0321
    im = load_image(image, 0, 0)
    if debug: print("Loaded image")
    import time
    start = time.time()
    ret = detect_image(net, meta, im, thresh, hier_thresh, nms, debug)
    print("network time: ", time.time() - start)
    free_image(im)
    if debug: print("freed image")
    return ret


def detect_image(net,
                 meta,
                 im,
                 thresh=.5,
                 hier_thresh=.5,
                 nms=.45,
                 debug=False):
    #import cv2
    #custom_image_bgr = cv2.imread(image) # use: detect(,,imagePath,)
    #custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
    #custom_image = cv2.resize(custom_image,(lib.network_width(net), lib.network_height(net)), interpolation = cv2.INTER_LINEAR)
    #import scipy.misc
    #custom_image = scipy.misc.imread(image)
    #im, arr = array_to_image(custom_image)		# you should comment line below: free_image(im)
    num = c_int(0)
    if debug: print("Assigned num")
    pnum = pointer(num)
    if debug: print("Assigned pnum")
    predict_image(net, im)
    letter_box = 0
    #predict_image_letterbox(net, im)
    #letter_box = 1
    if debug: print("did prediction")
    #dets = get_network_boxes(net, custom_image_bgr.shape[1], custom_image_bgr.shape[0], thresh, hier_thresh, None, 0, pnum, letter_box) # OpenCV
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0,
                             pnum, letter_box)
    if debug: print("Got dets")
    num = pnum[0]
    if debug: print("got zeroth index of pnum")
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    if debug: print("did sort")
    res = []
    if debug: print("about to range")
    for j in range(num):
        if debug: print("Ranging on " + str(j) + " of " + str(num))
        if debug: print("Classes: " + str(meta), meta.classes, meta.names)
        for i in range(meta.classes):
            if debug:
                print("Class-ranging on " + str(i) + " of " +
                      str(meta.classes) + "= " + str(dets[j].prob[i]))
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                if debug:
                    print("Got bbox", b)
                    print(nameTag)
                    print(dets[j].prob[i])
                    print((b.x, b.y, b.w, b.h))
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    if debug: print("did range")
    res = sorted(res, key=lambda x: -x[1])
    if debug: print("did sort")
    free_detections(dets, num)
    if debug: print("freed detections")
    return res


netMain = None
metaMain = None
altNames = None


def performDetect(imagePath="data/dog.jpg",
                  thresh=0.25,
                  configPath=PREFIX + "/cfg/yolov4.cfg",
                  weightPath=PREFIX + "/yolov4.weights",
                  metaPath=PREFIX + "/cfg/coco.data",
                  showImage=False,
                  makeImageOnly=True,
                  initOnly=False):
    """
    Convenience function to handle the detection and returns of objects.

    Displaying bounding boxes requires libraries scikit-image and numpy

    Parameters
    ----------------
    imagePath: str
        Path to the image to evaluate. Raises ValueError if not found

    thresh: float (default= 0.25)
        The detection threshold

    configPath: str
        Path to the configuration file. Raises ValueError if not found

    weightPath: str
        Path to the weights file. Raises ValueError if not found

    metaPath: str
        Path to the data file. Raises ValueError if not found

    showImage: bool (default= True)
        Compute (and show) bounding boxes. Changes return.

    makeImageOnly: bool (default= False)
        If showImage is True, this won't actually *show* the image, but will create the array and return it.

    initOnly: bool (default= False)
        Only initialize globals. Don't actually run a prediction.

    Returns
    ----------------------


    When showImage is False, list of tuples like
        ('obj_label', confidence, (bounding_box_x_px, bounding_box_y_px, bounding_box_width_px, bounding_box_height_px))
        The X and Y coordinates are from the center of the bounding box. Subtract half the width or height to get the lower corner.

    Otherwise, a dict with
        {
            "detections": as above
            "image": a numpy array representing an image, compatible with scikit-image
            "caption": an image caption
        }
    """
    # Import the global variables. This lets us instance Darknet once, then just call performDetect() again without instancing again
    global metaMain, netMain, altNames  #pylint: disable=W0603
    assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath) + "`")
    if netMain is None:
        netMain = load_net_custom(configPath.encode("ascii"),
                                  weightPath.encode("ascii"), 0,
                                  1)  # batch size = 1
    if metaMain is None:
        metaMain = load_meta(metaPath.encode("ascii"))
    if altNames is None:
        # In Python 3, the metafile default access craps out on Windows (but not Linux)
        # Read the names file and create a list to feed to detect
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    if initOnly:
        print("Initialized detector")
        return None
    if not os.path.exists(imagePath):
        raise ValueError("Invalid image path `" + os.path.abspath(imagePath) +
                         "`")
    # Do the detection
    #detections = detect(netMain, metaMain, imagePath, thresh)	# if is used cv2.imread(image)
    import time
    start = time.time()
    detections = detect(netMain, metaMain, imagePath.encode("ascii"), thresh)
    print("detections time :", time.time() - start)
    if showImage:
        try:
            from skimage import io, draw
            import numpy as np
            image = io.imread(imagePath)
            print("*** " + str(len(detections)) +
                  " Results, color coded by confidence ***")
            imcaption = []
            for detection in detections:
                label = detection[0]
                confidence = detection[1]
                pstring = label + ": " + str(np.rint(100 * confidence)) + "%"
                imcaption.append(pstring)
                print(pstring)
                bounds = detection[2]
                shape = image.shape
                # x = shape[1]
                # xExtent = int(x * bounds[2] / 100)
                # y = shape[0]
                # yExtent = int(y * bounds[3] / 100)
                yExtent = int(bounds[3])
                xEntent = int(bounds[2])
                # Coordinates are around the center
                xCoord = int(bounds[0] - bounds[2] / 2)
                yCoord = int(bounds[1] - bounds[3] / 2)
                boundingBox = [[xCoord, yCoord], [xCoord, yCoord + yExtent],
                               [xCoord + xEntent, yCoord + yExtent],
                               [xCoord + xEntent, yCoord]]
                # Wiggle it around to make a 3px border
                rr, cc = draw.polygon_perimeter([x[1] for x in boundingBox],
                                                [x[0] for x in boundingBox],
                                                shape=shape)
                rr2, cc2 = draw.polygon_perimeter(
                    [x[1] + 1 for x in boundingBox],
                    [x[0] for x in boundingBox],
                    shape=shape)
                rr3, cc3 = draw.polygon_perimeter(
                    [x[1] - 1 for x in boundingBox],
                    [x[0] for x in boundingBox],
                    shape=shape)
                rr4, cc4 = draw.polygon_perimeter(
                    [x[1] for x in boundingBox],
                    [x[0] + 1 for x in boundingBox],
                    shape=shape)
                rr5, cc5 = draw.polygon_perimeter(
                    [x[1] for x in boundingBox],
                    [x[0] - 1 for x in boundingBox],
                    shape=shape)
                boxColor = (int(255 * (1 - (confidence**2))),
                            int(255 * (confidence**2)), 0)
                draw.set_color(image, (rr, cc), boxColor, alpha=0.8)
                draw.set_color(image, (rr2, cc2), boxColor, alpha=0.8)
                draw.set_color(image, (rr3, cc3), boxColor, alpha=0.8)
                draw.set_color(image, (rr4, cc4), boxColor, alpha=0.8)
                draw.set_color(image, (rr5, cc5), boxColor, alpha=0.8)
            if not makeImageOnly:
                io.imshow(image)
                io.show()
            detections = {
                "detections": detections,
                "image": image,
                "caption": "\n<br/>".join(imcaption)
            }
        except Exception as e:
            print("Unable to show image: " + str(e))
    return detections


def performBatchDetect(thresh=0.25,
                       configPath="./cfg/yolov4.cfg",
                       weightPath="yolov4.weights",
                       metaPath="./cfg/coco.data",
                       hier_thresh=.5,
                       nms=.45,
                       batch_size=3):
    import cv2
    import numpy as np
    # NB! Image sizes should be the same
    # You can change the images, yet, be sure that they have the same width and height
    img_samples = ['data/person.jpg', 'data/person.jpg', 'data/person.jpg']
    image_list = [cv2.imread(k) for k in img_samples]

    net = load_net_custom(configPath.encode('utf-8'),
                          weightPath.encode('utf-8'), 0, batch_size)
    meta = load_meta(metaPath.encode('utf-8'))
    pred_height, pred_width, c = image_list[0].shape
    net_width, net_height = (network_width(net), network_height(net))
    img_list = []
    for custom_image_bgr in image_list:
        custom_image = cv2.cvtColor(custom_image_bgr, cv2.COLOR_BGR2RGB)
        custom_image = cv2.resize(custom_image, (net_width, net_height),
                                  interpolation=cv2.INTER_NEAREST)
        custom_image = custom_image.transpose(2, 0, 1)
        img_list.append(custom_image)

    arr = np.concatenate(img_list, axis=0)
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(net_width, net_height, c, data)

    batch_dets = network_predict_batch(net, im, batch_size, pred_width,
                                       pred_height, thresh, hier_thresh, None,
                                       0, 0)
    batch_boxes = []
    batch_scores = []
    batch_classes = []
    for b in range(batch_size):
        num = batch_dets[b].num
        dets = batch_dets[b].dets
        if nms:
            do_nms_obj(dets, num, meta.classes, nms)
        boxes = []
        scores = []
        classes = []
        for i in range(num):
            det = dets[i]
            score = -1
            label = None
            for c in range(det.classes):
                p = det.prob[c]
                if p > score:
                    score = p
                    label = c
            if score > thresh:
                box = det.bbox
                left, top, right, bottom = map(
                    int, (box.x - box.w / 2, box.y - box.h / 2,
                          box.x + box.w / 2, box.y + box.h / 2))
                boxes.append((top, left, bottom, right))
                scores.append(score)
                classes.append(label)
                boxColor = (int(255 * (1 - (score**2))), int(255 * (score**2)),
                            0)
                cv2.rectangle(image_list[b], (left, top), (right, bottom),
                              boxColor, 2)
        cv2.imwrite(os.path.basename(img_samples[b]), image_list[b])

        batch_boxes.append(boxes)
        batch_scores.append(scores)
        batch_classes.append(classes)
    free_batch_detections(batch_dets, batch_size)
    return batch_boxes, batch_scores, batch_classes


" ---------------------------------------------------------------------------"
" Modifications to Darknet Detector "

import numpy as np
from collections import namedtuple
from enum import Enum
import cv2

# Bounding box coordinates (image pixels) for detected objects
Bbox = namedtuple('BBox',
                  field_names=[
                      'top_left_x', 'top_left_y', 'bottom_right_x',
                      'bottom_right_y'
                  ])

# Person centroid position in camera coordinates where x is right, y is down
# and z is forward
Centroid = namedtuple('Centroid', field_names=['x', 'y', 'z'])


# Color indicating if the social distancing is violated or not
class Color(Enum):
    RED = [255, 0, 0]
    GREEN = [0, 255, 0]


class Detection(object):
    '''
    Data class for storing object detection results:
    Arguments:
        class_name: str, object class name
        score: float, object detection confidence score between 0 and 1
        bbox: list, bounding box coordinates in pixels
            [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
        centroid: list, x,y,z coordinates of object in camera coordinates
    '''
    def __init__(self, score, bbox, centroid=None):
        self.score = score
        self.bbox = bbox
        self.centroid = centroid
        # proximity to the closest person in the scene (meters)
        self.proximity = np.inf
        self.distance_to_cam = 0  # distance to the camera
        # color indicating if the social distancing is
        # maintained or not (GREEN, RED)
        self.color = None
        self.instance_id = -1  # id assigned by temporal filter


class PersonDetector:
    def __init__(self):
        performDetect(initOnly=True)
        self.net = netMain
        self.meta = metaMain
        self.net_width = network_width(self.net)
        self.net_height = network_height(self.net)

    def detect(self, cv_img, thresh=.8, hier_thresh=.5, nms=.45):
        pred_height, pred_width, c = cv_img.shape
        x_scale = float(pred_width) / self.net_width
        y_scale = float(pred_height) / self.net_height

        cv_img = cv2.resize(cv_img, (self.net_width, self.net_height),
                            interpolation=cv2.INTER_NEAREST)
        cv_img = cv_img.transpose(2, 0, 1)
        arr = np.ascontiguousarray(cv_img.flat, dtype=np.float32) / 255.0
        data = arr.ctypes.data_as(POINTER(c_float))
        c_img = IMAGE(self.net_width, self.net_height, c, data)

        detections = detect_image(self.net,
                                  self.meta,
                                  c_img,
                                  thresh=thresh,
                                  hier_thresh=hier_thresh,
                                  nms=nms)

        # filter person class
        detections = [
            detection for detection in detections if detection[0] == 'person'
        ]
        res = []
        for index in range(len(detections)):
            _ = detections[index][0]
            score = detections[index][1]
            bounds = detections[index][2]

            yExtent = int(bounds[3] * y_scale)
            xEntent = int(bounds[2] * x_scale)
            # Coordinates are around the center
            top_left_x = max(0, int((bounds[0] - bounds[2] / 2) * x_scale))
            top_left_y = max(0, int((bounds[1] - bounds[3] / 2) * y_scale))
            bottom_right_x = min(pred_width, top_left_x + xEntent)
            bottom_right_y = min(pred_height, top_left_y + yExtent)
            bbox = Bbox(top_left_x, top_left_y, bottom_right_x, bottom_right_y)
            res.append(Detection(score, bbox))
        return res


if __name__ == "__main__":
    import time
    start = time.time()
    print(performDetect())
    print(time.time() - start)
