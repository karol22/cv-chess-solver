# Code from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
import cv2
import numpy as np
from matplotlib import pyplot as plt


source_window = 'Source image'
corners_window = 'Corners detected'
max_thresh = 255

vid = cv2.VideoCapture(0)

if(vid.isOpened() == False):
    print('Error opening video stream or file')

#while(vid.isOpened()):
#    ret, img = vid.read()
#    if ret == True:

img = cv2.imread('board.jpg')
points = set()
img = cv2.resize(img, (960, 540))
img = img[50:500,200:750]
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
for i in range(len(dst)):
    for j in range(len(dst[0])):
        if dst[i][j] >0.02*dst.max():
            points.add((int((i+40)/80), int((j+40)/80)))
print("points", len(points))
print(points)
for x, y in points:
  img[80 * x, 80 * y]=[0,0,255]  


plt.imshow(img), plt.show()
"""
        if(cv2.waitKey(25) & 0xFF == ord('q')):
            break
    else:
        break
# vid.release()
"""