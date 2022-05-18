from data import *
from augmentation import get_augmentation
from layers import MultiBoxLoss
from ssd import SSD, DataIterator

from metrics.BoundingBox import BoundingBox
from metrics.BoundingBoxes import BoundingBoxes
from metrics.Evaluator import *
from metrics.utils import *

from torch.utils.tensorboard import SummaryWriter
import os
import os.path as osp
import time
from datetime import datetime
from tqdm import tqdm

import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
from torchvision.ops import batched_nms


import numpy as np
from glob import glob
import argparse

import pdb

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Testing With Pytorch')
parser.add_argument('--dataset', default='digit', type=str, 
                    help='Dataset name')
parser.add_argument('--thres', default=0.1, type=float,
                    help='Threshold for bounding box')
parser.add_argument('--input', default='input/inference', type=str,
                    help='Inference file or folder')
parser.add_argument('--save', default='input/prediction', type=str,
                    help='Folder to save predictions')
parser.add_argument('--trained_model', default='weights/digit_best_224.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

args = parser.parse_args()

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda")

else:
    args.cuda = False
    args.device = torch.device("cpu")

def test():
    _, test_transforms = get_augmentation(args.dataset)
    if args.dataset == 'digit':
        cfg = digitcfg
    
    net = SSD(cfg, args.device)

    if args.cuda:
        cudnn.benchmark = True

    if args.trained_model:
        print('loading model {}...'.format(args.trained_model))
        net.load_weights(args.trained_model)
    else:
        print('Please provide checkpoint')
        exit(0)

    net = net.to(args.device)
    net = net.eval()

    files = []
    if osp.isdir(args.input):
        ext = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
        for et in ext:
            files += glob(osp.join(args.input, et))
    
    elif osp.isfile(args.input):
        files.append(args.input)
    else:
        raise Exception("args.input doesn't exist")

    for file in files:
        print(file)
        infobj = DigitInference(file, transform=test_transforms)
        input = infobj.pull_input()
        size = infobj.get_max_size()
        input = input.to(args.device)
        net.configure_prior(size)

        with torch.no_grad():
            out = net(input, test=True)
            detections = out.data

        box_container = BoundingBoxes()
        scale = torch.Tensor([size, size, size, size])
        for i in range(1, detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= args.thres:
                conf = detections[0, i, j, 0].cpu().item()
                pt = (detections[0, i, j, 1:]*scale).cpu()
                box_str = str(torch.round(pt).tolist())
                print(f'digit: {i - 1}, conf: {conf}, box: {box_str}')
                pred_box = BoundingBox(imageName=file,
                                        classId=i - 1,
                                        classConfidence=conf,
                                        x=pt[0],
                                        y=pt[1],
                                        w=pt[2],
                                        h=pt[3],
                                        typeCoordinates=CoordinatesType.Absolute,
                                        bbType=BBType.Detected,
                                        format=BBFormat.XYX2Y2,
                                        imgSize=(size, size))

                box_container.addBoundingBox(pred_box)
                j += 1

        raw_img = infobj.pull_image()
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
        bbox_img = box_container.drawAllBoundingBoxes(image=raw_img, imageName=file)
        if not osp.exists(args.save):
            os.makedirs(args.save)

        cv2.imshow("prediction", bbox_img)
        cv2.waitKey(0)
        save_file_name = osp.join(args.save, osp.basename(file))
        cv2.imwrite(save_file_name, bbox_img)

if __name__ == '__main__':
    test()