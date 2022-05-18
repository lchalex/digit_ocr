import torch
import torch.nn as nn
import torch.nn.functional as F

def multibox_head(in_channels, boxes, num_classes):
    assert len(in_channels) == len(boxes), \
        F"length of in_channels must match boxes, {len(in_channels)} != {len(boxes)}"

    loc_layers = []
    conf_layers = []
    for i, (channel, num_b) in enumerate(zip(in_channels, boxes)):
        loc_layers += [nn.Conv2d(channel, num_b * 4, kernel_size=1, padding=0)]
        conf_layers += [nn.Conv2d(channel, num_b * num_classes, kernel_size=1, padding=0)]
    
    return loc_layers, conf_layers
