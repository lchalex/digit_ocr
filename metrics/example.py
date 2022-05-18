###########################################################################################
#                                                                                         #
# This sample demonstrates:                                                               #
# * How to create your own bounding boxes (detections and ground truth) manually;         #
# * Ground truth bounding boxes are drawn in green and detected boxes are drawn in red;   #
# * Create objects of the class BoundingBoxes with your bounding boxes;                   #
# * Create images with detections and ground truth;                                       #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: May 24th 2018                                                 #
###########################################################################################

import sys
sys.path.append('..')

from metrics.BoundingBox import BoundingBox
from metrics.BoundingBoxes import BoundingBoxes
from metrics.Evaluator import *
from metrics.utils import *
import pdb

###########################
# Defining bounding boxes #
###########################
# Ground truth bounding boxes of 000001.jpg
gts = [
        [[]],
        [[]],
        [[]],
        [[10, 240, 200, 260, 0]],
        [[200, 100, 230, 130, 0]],
        [[10, 20, 30, 30, 0], [50, 100, 150, 120, 0]]
    ]

pds = [
        [[]],
        [[]],
        [[]],
        [[10, 240, 200, 260, 0]],
        [[200, 100, 230, 130, 0]],
        [[10, 20, 30, 30, 0], [50, 100, 150, 120, 0]]
    ]

myBoundingBoxes = BoundingBoxes()
for img_id, bboxes in enumerate(gts):
    for box in bboxes:
        if len(box) > 0:
            print(f'Add GT {box} to image {img_id}')
            gt_boundingBox = BoundingBox(imageName=img_id,
                                        classId=box[4],
                                        x=box[0],
                                        y=box[1],
                                        w=box[2],
                                        h=box[3],
                                        typeCoordinates=CoordinatesType.Absolute,
                                        bbType=BBType.GroundTruth,
                                        format=BBFormat.XYX2Y2,
                                        imgSize=(300, 300))
            
            myBoundingBoxes.addBoundingBox(gt_boundingBox)

for img_id, bboxes in enumerate(pds):
    for box in bboxes:
        if len(box) > 0:
            print(f'Add Pred {box} to image {img_id}')
            detected_boundingBox = BoundingBox(imageName=img_id,
                                                classId=box[4],
                                                classConfidence=0.9,
                                                x=box[0],
                                                y=box[1],
                                                w=box[2],
                                                h=box[3],
                                                typeCoordinates=CoordinatesType.Absolute,
                                                bbType=BBType.Detected,
                                                format=BBFormat.XYX2Y2,
                                                imgSize=(300, 300))
            
            myBoundingBoxes.addBoundingBox(detected_boundingBox)


evaluator = Evaluator()
metric = evaluator.GetPascalVOCMetrics(myBoundingBoxes, IOUThreshold=0.5, method=MethodAveragePrecision.EveryPointInterpolation)
for mc in metric:
    print(mc['AP'])