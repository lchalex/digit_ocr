import torch
from torch.autograd import Function
from torchvision.ops import nms
from .box_utils import decode

import pdb

class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, cfg, bkg_label, top_k, nms_thresh, device):
        self.num_classes = cfg['num_classes']
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.variance = [0.1, 0.2]
        self.device = device

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        loc_data = loc_data.to(self.device)
        conf_data = conf_data.to(self.device)
        prior_data = prior_data.to(self.device)
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            argmax = torch.argmax(conf_scores, 0)

            comb_boxes = torch.zeros((0, 4)).to(self.device)
            comb_scores = torch.zeros(0).to(self.device)
            comb_cl = torch.zeros(0, dtype=int).to(self.device)

            for cl in range(1, self.num_classes):
                c_mask = argmax == cl
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                v, idx = scores.sort(0)  # sort in ascending order
                idx = idx[-self.top_k:]  # select top_k
                top_boxes = boxes[idx]
                top_scores = scores[idx]
                comb_boxes = torch.vstack((comb_boxes, top_boxes))
                comb_scores = torch.cat((comb_scores, top_scores))
                comb_cl = torch.cat((comb_cl, torch.ones(len(top_scores), dtype=int).to(self.device) * cl))

            nms_ids = nms(comb_boxes, comb_scores, self.nms_thresh) # Assume all classes are the same
            best_boxes = comb_boxes[nms_ids]
            best_scores = comb_scores[nms_ids]
            best_cl = comb_cl[nms_ids]

            for cl in range(1, self.num_classes):
                ids = torch.where(best_cl == cl)[0]
                count = ids.size(0)
                output[i, cl, :count] = \
                    torch.cat((best_scores[ids].unsqueeze(1),
                               best_boxes[ids]), 1)

        return output
