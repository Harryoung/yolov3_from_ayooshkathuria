from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

def arg_parse():
    """
    Parse arguments to the detect module

    """
    parser = argparse.ArgumentParser(description="YOLO v3 Detection Module for Video")
    parser.add_argument("--video", dest='video', help="Path of the videofile",type=str)
    parser.add_argument("--det", dest='det', help=
    "Directory to store the detection results", default="det", type=str)
    parser.add_argument("--bs", dest="bs",help="Batch Size", default=1)
    parser.add_argument("--confidence", dest="confidence",help="Objectness Confidence threshold", default=0.5)
    parser.add_argument("--nms_thresh",dest="nms_thresh",help="NMS Threshold",default=0.4)
    parser.add_argument("--cfg",dest = "cfgfile",help="Path for the configure file",default="cfg/yolov3.cfg",type=str)
    parser.add_argument("--weights",dest="weightsfile", help="Paths for the weights file",default="yolov3.weights",type=str)
    parser.add_argument("--reso", dest="reso",help=
    "Input resolution of the network. Increse to increase accuracy, decrease to speed",default="320",type=str)

    return parser.parse_args()

def letterbox_img(img, inp_dim):
    """
    Resize image with unchanged aspect ratio.

    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    sc_fator = min(w/img_w, h/img_h)
    new_w = int(img_w * sc_fator)
    new_h = int(img_h * sc_fator)
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h-new_h)//2:(h-new_h)//2+new_h,(w-new_w)//2:(w-new_w)//2+new_w,:] = resized_image
    return canvas

def prep_image(img, inp_dim):
    """
    Preprocess image

    Returns a Variable
    """
    img = letterbox_img(img,(inp_dim, inp_dim))
    img = img[:,:,::-1].transpose(2,0,1).copy()  # Why copy()? from_numpy: The returned tensor and `ndarray` share the same memory. 
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

colors = pkl.load(open("pallete", "rb"))

def draw_rectangle(x, frame):
    """
    Draw rectangle and label on image

    """
    color = random.choice(colors)
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = frame
    cls = int(x[-1])
    label = "{0}  {1:5.2f}".format(classes[cls], float(x[-2]))
    line_width = int((img.shape[0] + img.shape[1]) / 2 / 600)
    cv2.rectangle(img, c1, c2, color, line_width)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, line_width, line_width)[0]
    c2 = c1[0] + t_size[0] +3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, line_width, [255,255,255],line_width)
    return img


args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
# start = 0
CUDA = torch.cuda.is_available()

def load_classes(namesfile):
    fp = open(namesfile,"r")
    names = fp.read().split('\n')[:-1]
    return names

num_classes = 80  # For COCO dataset
classes = load_classes("data/coco.names")

# Set up the neural network
print("Loading network......")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

if CUDA:
    model.cuda()

model.eval()

# Detection phase
# Load the video
videofile = args.video
# cap = cv2.VideoCapture(videofile)
print("before open webcam")
cap = cv2.VideoCapture(0)     # for webcam
print("after open webcam")
assert cap.isOpened(), 'Cannot capture source'

if not osp.exists(args.det):
    os.makedirs(args.det)

# Iterate over frames
frames = 0
start = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        img = prep_image(frame, inp_dim)
        im_dim_list = frame.shape[1], frame.shape[0]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

        if CUDA:
            im_dim_list = im_dim_list.cuda()
            img = img.cuda()
        
        with torch.no_grad():
            img_var = Variable(img)
        output = model(img_var, CUDA)
        output = write_results(output, confidence, num_classes, nms_conf=nms_thresh)

        if type(output) == int:  # No object detected
            frames += 1
            print("FPS of the video is {:5.4f}. No object is detected.".format(frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue
        # Compute the scaling factor
        im_dim_list = im_dim_list.index_select(0, output[:,0].long())
        scaling_factor = torch.min(inp_dim/im_dim_list, 1)[0].view(-1,1)
        # Transform from scaled letterbox coordinates to original images' coordinates
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
        output[:,1:5] /= scaling_factor
        # Clip the bounding boxes which go beyond the original images
        for i in range(output.shape[0]):
            output[i,[1,3]] = torch.clamp(output[i,[1,3]], 0.0, im_dim_list[i,0])
            output[i,[2,4]] = torch.clamp(output[i,[2,4]], 0.0, im_dim_list[i,1])

        list(map(lambda x: draw_rectangle(x, frame), output))

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        print("FPS of the video is {:5.4f}".format(frames / (time.time() - start)))
    else:
        break

torch.cuda.empty_cache()

