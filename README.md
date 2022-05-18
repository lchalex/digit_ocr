# Digit OCR
Anchor based detection model  
The repository is modified from [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)  

## Dependency
 - tqdm
 - cv2
 - numpy
 - pandas
 - torch
 - torchvision
 - albumentations

## Dataset
 - MNIST

Validation Samples
<p align="left">
    <img src="doc/val_1.png" width="224" title="valid image 1"/>
    <img src="doc/val_2.png" width="224" title="valid image 2"/>
    <img src="doc/val_3.png" width="224" title="valid image 3"/>
    <img src="doc/val_4.png" width="224" title="valid image 4"/>
</p>

## Augmentation
### Training
 - random affine
 - random threshold
 - random scaling
 - random noise
 - random erode / dilate
 - color normalization

### Testing
 - pad to square
 - color normalization

## Architecture
 - Resnet backbone + FPN neck + SSD head
<p align="center">
    <img src="doc/archi.png" title="Architecture" />
    <img src="doc/ssd_head.png" height="200" title="Head" />
    <img src="doc/bottleneck.png" height="200" title="Block" />
</p>  
  

## Anchor boxes
 - 4 boxes per feature map pixel
 - Total 16600 boxes (56x56x4 + 28x28x4 + 14x14x4 + 7x7x4) for 224x224 input
  
(See data/config.py)  
<p align="left">
    <img src="doc/anchor.png" title="anchor boxes">
</p>

## Loss Function
 - localization (smooth L1) + confidence (cross entropy) loss
 - Hard negative mining with ratio of 1:3
 - Postive samples: IoU >= 0.5, netural samples: 0.5 > IoU > 0.1

## Post-processing
 - Non maximum suppression (IoU > 0.1, ignore class id)

## Inference Results
<p align="left">
    <img src="doc/pred_1.png" title="pred image 1">
    <img src="doc/pred_2.png" title="pred image 2">
    <img src="doc/pred_3.png" title="pred image 3">
    <img src="doc/pred_4.png" title="pred image 4">
</p>

