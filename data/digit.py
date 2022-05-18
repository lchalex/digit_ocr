import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import math
import pandas as pd
import json
import random
from copy import deepcopy, copy
from glob import glob

def get_digit_image_path(root):
    train = glob(osp.join(root, "train", "*", "*.jpg"))
    test = glob(osp.join(root, "test", "*", "*.jpg"))
    return train, test

def random_place(white, digit_image, label, aug=False, seed=False):
    # Augmentation
    if seed:
        random.seed(str(digit_image.flatten().tolist()))

    if aug:
        if random.random() < 0.25:
            ar = np.clip(np.random.normal(1.0, 0.5), 0, 2)
            if ar < 1:
                ar = 1 / (2 - ar)
            digit_image = cv2.resize(digit_image, (int(digit_image.shape[1] * math.sqrt(ar)), int(digit_image.shape[0] / math.sqrt(ar))), interpolation=cv2.INTER_AREA)

        scale = 1
        if random.random() < 0.5:
            max_scale = min(white.shape[0] / digit_image.shape[0], white.shape[1] / digit_image.shape[1])
            scale = random.uniform(1, max(max_scale, 1))
            digit_image = cv2.resize(digit_image, (int(digit_image.shape[0] * scale), int(digit_image.shape[1] * scale)), interpolation=cv2.INTER_AREA)
        
        if random.random() < 0.5:
            _, digit_image = cv2.threshold(digit_image, 127, 255, cv2.THRESH_BINARY)
        
        if scale >= 3:
            if random.random() < 0.5:
                if random.random() < 0.5:
                    digit_image = cv2.erode(digit_image, np.ones((3, 3), np.uint8), iterations=random.randint(1, int(scale) // 3))
                else:
                    digit_image = cv2.dilate(digit_image, np.ones((3, 3), np.uint8), iterations=random.randint(1, int(scale) // 3))
            
    d_height, d_width = digit_image.shape
    w_height, w_width = white.shape
    ymin, xmin = random.randint(0, w_height - d_height), random.randint(0, w_width - d_width)
    white[ymin: ymin + d_height, xmin: xmin + d_width] = digit_image
    if aug:
        if random.random() < 0.2: # random noise
            for _ in range(random.randint(1, 3)):
                y = random.randint(0, w_height - 1)
                x = random.randint(0, w_width - 1)
                white[y,x] = 0

        if random.random() < 0.5: # invert color
            white = 255 - white
            
    return white, [[xmin / w_width, ymin / w_height, (xmin + d_width) / w_width, (ymin + d_height) / w_height, label]]

class DigitDataset(data.Dataset):
    def __init__(self, images_path, partition, size=640, transform=None):
        self.images_path = images_path
        self.partition = partition
        self.size = size
        self.transform = transform

    def __getitem__(self, index):
        im, gt = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.images_path)

    def pull_item(self, index):
        if not osp.isfile(self.images_path[index]):
            print(self.images_path[index], ' does not exist.')
            exit(0)

        img, target = self.pull_image(index)
        if self.transform is not None:
            if len(target) == 0:
                transformed = self.transform(image=img, bboxes=[[]])
                img = transformed['image']
                target = np.array([])
            else:
                target = np.array(target)
                transformed = self.transform(image=img, bboxes=target)
                img = transformed['image']
                target = transformed['bboxes']

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        return torch.from_numpy(img).permute(2, 0, 1), target

    def pull_image(self, index):
        digit_image = cv2.imread(self.images_path[index], cv2.IMREAD_GRAYSCALE)
        label = int(self.images_path[index].split(os.sep)[-2])
        white = np.ones((self.size, self.size), dtype=np.uint8) * 255
        if self.partition == "train":
            img, target = random_place(white, digit_image, label, aug=True, seed=False)
        else:
            img, target = random_place(white, digit_image, label, aug=True, seed=True)

        return img, target

class DigitInference(object):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        self.img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        self.height, self.width = self.img.shape
        self.max_size = max(self.height, self.width)
        self.pad_to_sqaure()

    def pull_image(self):
        return self.img.copy()
    
    def pull_padded(self):
        return self.padded.copy()

    def pull_input(self):
        img = self.pull_padded()
        img = self.transform(image=img, bboxes=[[0.0, 0.0, 1.0, 1.0, 0]])['image']
        img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)
        tensor = torch.from_numpy(img).permute(0, 3, 1, 2)
        return tensor

    def get_raw_size(self):
        return (self.height, self.width)

    def get_max_size(self):
        return self.max_size

    def pad_to_sqaure(self):
        padded = np.ones((self.max_size, self.max_size), dtype=np.uint8) * 255
        padded[:self.height, :self.width] = self.img.copy()
        self.padded = padded
