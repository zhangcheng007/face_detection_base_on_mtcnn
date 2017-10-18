#coding:utf-8
import sys
sys.path.append('..')
from detection.MtcnnDetector import MtcnnDetector
from detection.detector import PDetector,RODetector
from models.mtcnnmodel import P_Net, R_Net, O_Net
from utils.loader import TestLoader
import cv2
import os
import numpy as np
test_mode = "ONet"
thresh = [0.9, 0.6, 0.7]
min_face_size = 25
stride = 2
slide_window = False
shuffle = False
detectors = [None, None, None]
prefix = ['../models_check/PNet', '../models_check/RNet', '../models_check/ONet']
epoch = [18, 14, 16]
batch_size = [2048, 256, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
# load pnet model
if slide_window:
    PNet = PDetector(P_Net, 12, batch_size[0], model_path[0])
else:
    PNet = PDetector(P_Net, model_path[0])
detectors[0] = PNet

# load rnet model
RNet = RODetector(R_Net, 24, batch_size[1], model_path[1])
detectors[1] = RNet

# load onet model
ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)
gt_imdb = []

path = "xixi"
for item in os.listdir(path):
    gt_imdb.append(os.path.join(path,item))

test_data = TestLoader(gt_imdb)

all_boxes,landmarks = mtcnn_detector.detect_face(test_data)

count = 0
for imagepath in gt_imdb:
    image = cv2.imread(imagepath)
    for bbox in all_boxes[count]:
        cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))      
    print (imagepath)
    count = count + 1
    cv2.imwrite("{}.jpg".format(count), image)
    #cv2.imwrite("result_landmark/%d.png" %(count),image)
    cv2.imshow("lala",image)
    cv2.waitKey(0)  