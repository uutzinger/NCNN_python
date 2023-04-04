from utils import drawFaceObjects
import cv2
import time
from blazeface import BlazeFace

# img = cv2.imread('Urs.jpg')
# img = cv2.imread('Angelina.jpg')
# img = cv2.imread('Pic_Team1.jpg')
# img = cv2.imread('Pic_Team2.jpg')
# img = cv2.imread('Pic_Team3.jpg')
img = cv2.imread('worlds-largest-selfie.jpg') # 23ms
if img is None: print('Error opening image')

# find faces, low threshold shows performance of algorithm, make it large for real world application
net = BlazeFace(prob_threshold=0.8, nms_threshold=0.3, use_gpu=True)

tic = time.perf_counter()
for i in range(100):
    faceobjects = net(img, scale=True)
toc = time.perf_counter()
print((toc-tic)/100.)

drawFaceObjects(img, faceobjects)
cv2.imshow('Faces Blaze', img)    
cv2.waitKey()
