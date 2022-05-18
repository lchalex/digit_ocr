import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from models.encoder import get_encoder
from models.head import multibox_head
from models.decoder import FPNDecoder
import os

class DataIterator(object):
    def __init__(self, loader):
        assert isinstance(loader, torch.utils.data.DataLoader), 'Wrong loader type'
        self.loader = loader
        self.iterator = iter(self.loader)

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)

        return batch

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base Xception network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        cfg: (dict) config
    """

    def __init__(self, cfg, device):
        super(SSD, self).__init__()
        self.cfg = cfg
        self.device = device
        self.num_classes = self.cfg['num_classes']
        self.encoder_name = self.cfg['backbone']

        # SSD network
        self.backbone = get_encoder(self.encoder_name, pretrained=True)
        backbone_out_channels = self.backbone.get_out_channels()

        self.decoder = FPNDecoder(encoder_channels=list(backbone_out_channels))
        fpn_out_channels = self.decoder.get_out_channels()

        head_in_channels = fpn_out_channels
        num_boxes = [2 + 2 * len(x) for x in self.cfg['aspect_ratios']]
        head = multibox_head(head_in_channels, num_boxes, self.num_classes)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect(self.cfg, 0, 100, 0.1, self.device)

    def configure_prior(self, size):
        self.cfg['size'] = size
        in_channels = self.backbone._in_channels
        x = torch.rand(1, in_channels, size, size).to(self.device)
        with torch.no_grad():
            enc = self.backbone(x)
        
        self.cfg['feature_maps'] = [m.shape[-1] for m in enc[::-1]]

        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), requires_grad=True)

    def forward(self, x, test=False):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,cfg["size"],cfg["size"]].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        features = list()
        loc = list()
        conf = list()

        backbone_out = self.backbone(x)
        for i in range(len(backbone_out)):
            x = backbone_out[i]
            features.append(x)

        sources = self.decoder(*features)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if test:
            output = self.detect.forward(
                loc.view(loc.size(0), -1, 4),                                 # loc preds
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data))                                # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
