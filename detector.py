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
    parser = argparse.ArgumentParser(description="YOLO v3 Detection Module for Images")
    parser.add_argument("--images", dest='images', help=
    "Image/Directory containing images to perform detection upon", default='imgs', type=str)
    parser.add_argument("--det", dest='det', help=
    "Directory to store the detection results", default="det", type=str)
    parser.add_argument("--bs", dest="bs",help="Batch Size", default=1)
    parser.add_argument("--confidence", dest="confidence",help="Objectness Confidence threshold", default=0.5)
    parser.add_argument("--nms_thresh",dest="nms_thresh",help="NMS Threshold",default=0.4)
    parser.add_argument("--cfg",dest = "cfgfile",help="Path for the configure file",default="cfg/yolov3.cfg",type=str)
    parser.add_argument("--weights",dest="weightsfile", help="Paths for the weights file",default="yolov3.weights",type=str)
    parser.add_argument("--reso", dest="reso",help=
    "Input resolution of the network. Increse to increase accuracy, decrease to speed",default="608",type=str)

    return parser.parse_args()

args = arg_parse()
images = args.images
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
read_dir = time.time()
try:
    imlist = [osp.join(images,img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(images)
except FileNotFoundError:
    print("No file or directory with the name {}".format(images))
    exit()

if not osp.exists(args.det):
    os.makedirs(args.det)

load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]

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

im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

# Store the original size of images
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
if CUDA:
    im_dim_list = im_dim_list.cuda()

# Create batches
leftover = 0
if (len(imlist) % batch_size):
    leftover = 1
if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    im_batches = [torch.cat((im_batches[i*batch_size:min((i+1)*batch_size, len(imlist))])) for i in range(num_batches)]

# Iterate over batches
write = 0
start_det_loop = time.time()
for i, batch in enumerate(im_batches):
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    with torch.no_grad():
        input_var = Variable(batch)
    prediction = model(input_var, CUDA)
    prediction = write_results(prediction, confidence,num_classes,nms_conf = nms_thresh)
    end = time.time()

    if type(prediction) == int:  # No object detected
        for im_num, image in enumerate(imlist[i*batch_size:min((i+1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end-start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("--------------------------------------------------------------------------")
        continue

    prediction[:,0] += i*batch_size  # Transform form index in batch to index in image list

    if not write:
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))
    
    for im_num, image in enumerate(imlist[i*batch_size:min((i+1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end-start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("--------------------------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()  # Ensure syn call, which GPU return control to CPU exactly when FINISH the compution, not when START to compute.
        
try:
    output
except NameError:  # Not initialized
    print("No Objects were detected.")
    exit()

output_recast = time.time()
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

color_load = time.time()
colors = pkl.load(open("pallete", "rb"))

draw = time.time()
def draw_rectangle(x, results):
    """
    Draw rectangle and label on image

    """
    color = random.choice(colors)
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    line_width = int((img.shape[0] + img.shape[1]) / 2 / 400)
    font_scale = int((img.shape[0] + img.shape[1]) / 2 / 200) * (c2[1]-c1[1]) / img.shape[1]
    font_line_thickness = font_scale
    cv2.rectangle(img, c1, c2, color, line_width)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, font_scale, font_line_thickness)[0]
    c2 = c1[0] + t_size[0] +3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, font_scale, [255,255,255],font_line_thickness)
    return img

list(map(lambda x: draw_rectangle(x, loaded_ims), output))
det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, x.split("/")[-1]))
list(map(cv2.imwrite, det_names, loaded_ims))  # If without list, images will not be written to disk
end = time.time()

# Print Time Summary
print("SUMMARY")
print("-------------------------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output processing", color_load - output_recast))
print("{:25s}: {:2.3f}".format("Loading colors and drawing boxes", end - color_load))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/ len(imlist)))
print("-------------------------------------------------------------------------")

torch.cuda.empty_cache()

