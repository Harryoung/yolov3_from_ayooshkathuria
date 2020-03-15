from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    """
    Transforms the yolo module output into easy-reading prediction tensor.

    """

    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_szie = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_szie*grid_szie)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_szie*grid_szie*num_anchors, bbox_attrs)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    # anchors = [(a[0]*inp_dim/416/stride, a[1]*inp_dim/416/stride) for a in anchors] # WRONG!!! The clustered anchors are corresponding to 416, but should keep unchanged when input_dim varies.
    # Sigmoid the Centre_X, Centre_Y and objectness
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # Add the centre offsets
    grid = np.arange(grid_szie)
    a, b = np.meshgrid(grid, grid)
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)
    prediction[:,:,:2] += x_y_offset

    # Apply the exp hight and width
    anchors = torch.FloatTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_szie*grid_szie, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    # Apply sigmoid activation to class score
    prediction[:,:,5:5+num_classes] = torch.sigmoid(prediction[:,:,5:5+num_classes])

    # Resize the detection map to the size of input image
    prediction[:,:,:4] *= stride

    return prediction

def unique(tensor):
    """
    Returns a new tensor that contains unque values in tensor.

    """
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape) # Why new? For CUDA tensors, this method will create new tensor on the same device as this tensor.
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(box1, box2):
    """
    Returns the IoUs of two groups of bounding boxes. Here box1 contains only one box.

    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # Get the coordinates of intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.max(b1_x2, b2_x2)
    inter_rect_y2 = torch.max(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min = 0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min = 0)

    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    """
    Returns a D*8 tensor. D is for true detection bbox. 8 cols consist of batch_index, LT_x, LT_Y, RB_x, RB_y, objectness, class confidence, and class index

    """
    # Set the low objectness rows to zero
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask
    # Transform from (CenterX, CenterY, Width, Height) to (LT_x, LT_y, RB_x, RB_y)
    box_corner = prediction.new(prediction.shape) # A little bit wasted
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    # Tranform the conditional class prob to class confidence (added by YJB)
    objectness = prediction[:,:,4].unsqueeze(2)
    prediction[5:] = prediction[5:] * objectness

    batch_size = prediction.size(0)
    # Iterate through images in a batch
    write = 0
    for ind in range(batch_size):
        image_pred = prediction[ind]
        max_conf, max_conf_index = torch.max(image_pred[:,5:], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_index = max_conf_index.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_index)
        image_pred = torch.cat(seq, 1)

        # Get rid of the zeroed rows
        non_zero_ind = torch.nonzero(image_pred[:,4])
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1, 7) # I don't think view(-1,7) is necessary
        except: # For the case that no rows left. Oh! The view(-1,7) maybe used to throw the exception!
            continue
        # If newer version of torch do not throw the exception, we need an extra judge to be compatible
        if image_pred_.shape[0] == 0:
            continue
        # Get the uniqe class in an image
        img_classes = unique(image_pred_[:,-1])
        
        # Iterate through the classes
        for cls in img_classes:
            # Get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # Sort the rows by the class confidence
            conf_sort_index = torch.sort(image_pred_class[:,5],descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]

            idx = image_pred_class.size(0)
            # NMS
            for i in range(idx):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                # image_pred_class will lose rows during the loop, so maybe index go beyond the real size.
                except ValueError:
                    break
                except IndexError:
                    break

                # Zero out all the detections that have IoU > threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                # Get rid of the nonzero rows
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)
            # Add the batch index to result
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = 1
            else:
                out = torch.cat(seq ,1)
                output = torch.cat((output, out))
    
    try:
        return output
    except: # Case that there is not any entry left
        return 0


