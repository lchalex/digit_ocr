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
from torchvision.transforms import transforms

from glob import glob
import cv2
import numpy as np
from collections import Counter
import argparse

import pdb

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='digit', type=str,
                    help='Dataset root')
parser.add_argument('--dataset_root', default='./input',
                    help='Dataset root directory path')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--thres', default=0.1, type=float,
                    help='Threshold for bounding box')
parser.add_argument('--basenet', default=None, type=str,
                    help='Backbone state_dict file to resume training from')
parser.add_argument('--resume', default=None, type=str,
                    help='Whole network state_dict file to resume training from')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-5, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if torch.cuda.is_available() and args.cuda:
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda")
    
else:
    args.device = torch.device("cpu")
    args.cuda = False

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def train():
    train_transforms, valid_transforms = get_augmentation(args.dataset)

    if args.dataset == 'digit':
        cfg = digitcfg
        trainpath, testpath = get_digit_image_path(args.dataset_root)
        trainset = DigitDataset(trainpath, 'train', size=cfg['size'], transform=train_transforms)
        validset = DigitDataset(testpath, 'test', size=cfg['size'], transform=valid_transforms)

    else:
        raise 

    cfg['lr'] = args.lr
    cfg['bacth_size'] = args.batch_size
    cfg['weight_decay'] = args.weight_decay

    print('train/val size = {}/{}'.format(len(trainset), len(validset)))

    ssd_net = SSD(cfg, args.device)
    net = ssd_net

    if args.cuda:
        net = ssd_net
        cudnn.benchmark = True

    if args.basenet:
        print('Load basenet from {}...'.format(args.basenet))
        ssd_net.backbone.load_state_dict(torch.load(args.basenet))
    else:
        ssd_net.backbone.apply(weights_init)

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
        ssd_net.decoder.apply(weights_init)

    if args.cuda:
        net = net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(cfg['lr_steps']), gamma=0.1)

    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, 0, cfg['np_ratio'], args.device)

    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(trainset) // args.batch_size
    print('Training SSD on:', args.dataset)
    print('Using the specified args:')
    print(args)
    num_param = sum(p.numel() for p in net.parameters())
    print(f'Number of parameters = {num_param}')

    train_loader = data.DataLoader(trainset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    valid_loader = data.DataLoader(validset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False, collate_fn=detection_collate,
                                  pin_memory=True)

    timestr = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    writer = SummaryWriter(osp.join('runs', args.dataset + '_' + timestr))
    writer.add_text('cfg', str(cfg))

    batch_iterator = DataIterator(train_loader)
    iterations = cfg['images_per_epoch'] // args.batch_size
    best_ap = 0

    for epoch in range(cfg['epoch']):
        net.train()
        num_box = 0
        loc_loss = 0
        conf_loss = 0
        pbar = tqdm(range(1, iterations + 1))
        for iter_count in pbar:
            # load train data
            images, targets = next(batch_iterator)
            if sum([len(t) for t in targets]) == 0:
                continue

            if args.cuda:
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]
            else:
                targets = [ann for ann in targets]

            # forward
            out = net(images)
            soft = ssd_net.softmax(out[1])
            num_box += torch.sum(torch.argmax(soft, 2) != 0).item() / images.size(0)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            pbar.set_postfix(**{'loss_l': loc_loss / iter_count, 'loss_c': conf_loss / iter_count, 'num_box': num_box / iter_count, 'lr': optimizer.param_groups[0]['lr']})


        scheduler.step()
        writer.add_scalar('train/loss_l', loc_loss / iter_count, epoch)
        writer.add_scalar('train/loss_c', conf_loss / iter_count, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        net.eval()
        box_container = BoundingBoxes()
        pbar = tqdm(valid_loader)
        for batch_num, (images, targets) in enumerate(pbar):
            if args.cuda:
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]
            else:
                targets = [ann for ann in targets]

            with torch.no_grad():
                out = net(images, test=True)
                detections = out.data

            scale = torch.Tensor([cfg["size"], cfg["size"], cfg["size"], cfg["size"]])
            for cur in range(len(targets)):
                for box in targets[cur]:
                    gt_box = BoundingBox(imageName=batch_num * args.batch_size + cur,
                                        classId=int(box[-1]),
                                        x=int(box[0].item() * cfg["size"]),
                                        y=int(box[1].item() * cfg["size"]),
                                        w=int(box[2].item() * cfg["size"]),
                                        h=int(box[3].item() * cfg["size"]),
                                        typeCoordinates=CoordinatesType.Absolute,
                                        bbType=BBType.GroundTruth,
                                        format=BBFormat.XYX2Y2,
                                        imgSize=(cfg["size"], cfg["size"]))
                    
                    box_container.addBoundingBox(gt_box)

                for i in range(1, detections.size(1)):
                    j = 0
                    while detections[cur, i, j, 0] >= args.thres:
                        pt = (detections[cur, i, j, 1:]*scale).cpu().numpy()
                        pred_box = BoundingBox(imageName=batch_num * args.batch_size + cur,
                                                classId=i - 1,
                                                classConfidence=detections[cur, i, j, 0].item(),
                                                x=int(pt[0]),
                                                y=int(pt[1]),
                                                w=int(pt[2]),
                                                h=int(pt[3]),
                                                typeCoordinates=CoordinatesType.Absolute,
                                                bbType=BBType.Detected,
                                                format=BBFormat.XYX2Y2,
                                                imgSize=(cfg["size"], cfg["size"]))

                        box_container.addBoundingBox(pred_box)
                        j += 1
                        if j >= 100:
                            break

        evaluator = Evaluator()
        valid_metric = evaluator.GetPascalVOCMetrics(box_container, IOUThreshold=0.5, method=MethodAveragePrecision.EveryPointInterpolation)
        class_ap = []
        for mc in valid_metric:
            class_ap.append(mc['AP'])

        avg_ap = sum(class_ap) / len(class_ap)
        print('Valid avg AP = ', avg_ap)
        writer.add_scalar('valid/avg_AP', avg_ap, epoch)

        if avg_ap > best_ap:
            best_ap = avg_ap
            print('Saving state, epoch:', epoch)
            torch.save(ssd_net.state_dict(), args.save_folder + args.dataset + '_' +
                        repr(epoch) + '.pth')

            for idx in range(0, len(validset), max(len(validset) // 200, 1)):
                img, _ = validset.pull_image(idx)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                bbox_img = box_container.drawAllBoundingBoxes(image=img, imageName=idx)
                cv2.imwrite(osp.join('log_image', str(idx).zfill(5) + '.png'), bbox_img)

        net.train()

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)

if __name__ == '__main__':
    start_t = time.time()
    train()
    print('Elapsed time', (time.time() - start_t)/60, 'mins')
