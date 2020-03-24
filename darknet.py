from __future__ import division

from util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def parse_cfg(cfgfile):
    """
    parse the configuration file.

    Returns a list of blocks. Each blocks describes a block in the neural 
    network to be built. Block is represented as a dictionary in the list.
    
    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x)>0 and x[0]!='#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def create_modules(blocks):
    net_info = blocks[0]    # infomation about the input and pre-processing
    module_list = nn.ModuleList()
    pre_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        
        # check the type of the block
        # create a new module for the block
        # append to module_lsit

        if (x["type"] == "convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            
            # Add the convolutional layer
            conv = nn.Conv2d(pre_filters,filters,kernel_size,stride,pad,bias=bias)
            module.add_module("conv_{0}".format(index), conv)
            # Add the Batch Norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)
            # Add the activation layer
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)
        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode = "bilinear",align_corners=False)
            module.add_module("upsample_{0}".format(index), upsample)
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        elif (x["type"] == "shortcut"):
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(index), shortcut)
        elif (x["type"] == "yolo"):
            mask = x["mask"].split(',')
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{0}".format(index), detection)
        # print(index, module)
        module_list.append(module)
        pre_filters = filters
        output_filters.append(filters)
    return (net_info, module_list)

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}    # Cache the outputs for route layers

        write = 0    # Flag for if we have got the first scale output feature map
        for i, module in enumerate(modules):
            # print(i,module)
            module_type = module["type"]
            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                
                if (layers[0] > 0):
                    layers[0] = layers[0] - i
                
                if len(layers) == 1:
                    x = outputs[i + layers[0]]

                else:
                    if(layers[1]) > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)  # B*C*H*W, 1 for C
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors

                # Get the input dimensions
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(module["classes"])

                # Transform
                x = x.data  # Transform from Variable to Tensor
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
                
            outputs[i] = x
        return detections
        
    def load_weights(self, weight_file):
        # Open the weight file
        fp = open(weight_file, "rb")

        # The first 3 int32 and 1 int64 values are header information
        # 1. Major version number
        # 2. Minor version number
        # 3. Subversion number
        # 4. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 3)
        self.header = torch.from_numpy(header)
        seen = np.fromfile(fp, dtype = np.int64, count = 1)
        self.seen = torch.from_numpy(seen)

        weights = np.fromfile(fp, dtype = np.float32)

        # Iterate over the weights file
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]

            # If module_type is convolutional, load the weights. Otherwise ignore.
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize): # If batch_normalize is True, load the weights for Batch Norm
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)  # β
                    bn_weights = bn_weights.view_as(bn.weight.data)  # α
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)  # tensor, not trainable
                    bn_running_var = bn_running_var.view_as(bn.running_var)  # tensor, not trainable

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else: # If False, load the biases for conv layer
                    # Get the number of biases of conv
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases])
                    ptr += num_biases
                    # Cast the loaded weights into dims of model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    # Copy the weights to model
                    conv.bias.data.copy_(conv_biases)

                # Now load the weights for Conv Layers
                num_weights = conv.weight.numel()
                # Do the same as above
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (608, 608))
    img_ = img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_

if __name__ == '__main__':
    # blocks = parse_cfg('cfg/yolov3.cfg')
    # print(blocks)
    # net_info, module_list = create_modules(blocks)
    
    model = Darknet('cfg/yolov3.cfg')
    model.load_weights('yolov3.weights')
    # inp = get_test_input()
    # pred = model(inp, torch.cuda.is_available())
    # print(pred)

