 # Code from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
import cv2
import numpy as np


source_window = 'Source image'
corners_window = 'Corners detected'




vid = cv2.VideoCapture(0)

if(vid.isOpened() == False):
    print('Error opening video stream or file')

while(vid.isOpened()):
    ret, img = vid.read()
    if ret == True:
        img = img[100:1000, 600:1500]
        cv2.namedWindow("image", cv2.WINDOW_NORMAL);
        cv2.imshow('image', img)
        if(cv2.waitKey(25) & 0xFF == ord('q')):
            break
    else:
        break
vid.release()