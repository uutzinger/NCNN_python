# NCNN Python

PRERELEASE STATUS, MIGHT HAVE ERRORS

- [NCNN Python](#ncnn-python)
  * [Overview](#overview)
    + [Acceleration](#acceleration)
  * [Requirements](#requirements)
  * [Examples & Models](#examples---models)
  * [Installation](#installation)
    + [To install OpenCV on Raspi:](#to-install-opencv-on-raspi-)
  * [Run Example](#run-example)
    + [Example Programs](#example-programs)
  * [Pip upload](#pip-upload)
  * [Changes](#changes)
  * [References](#references)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

## Overview
This is a collection of python implementations of NCNN examples. Code was converted to python and accelerated with numpy and cython. A basic frame work for Points, Rects, Objects, Non Maximum Supression and image manipulation was implemented. In general this does not accelerate the examples but provides an optimized python version.

There are several sites listing current implementation of CNN models in excess of the ones listed by [Tencent](https://github.com/Tencent/ncnn/blob/master/python/ncnn/model_zoo). However, there is no single authorative repository.
* [Baiyuetribe
](https://github.com/Baiyuetribe/ncnn-models)
* [Marton Juhasz](https://github.com/nilseuropa/ncnn_models)

When converting code to python, torch dependencies were removed. A useful approaches is listed here: [torch to numpy](https://medium.com/axinc-ai/conversion-between-torch-and-numpy-operators-ce189b3882b1) guideline

To increase performance, the following rules were observed:
- Numpy boolean indexing is faster, avoid ```np.where```
- ```np.append```, ```np.vstack``` or ```np.stack``` should be avoided when addign small amounts of data and regular lists and append function can be used
- If ```np.comcatenate``` is needed, apply it once to a list of all objects that need concatenation
- If ```np.max``` and ```np.argmax``` are needed in 3D array use programs shown below
- OpenCV data manipulations are faster than Numpy
- OpenCV is slighlty faster than NCCN when manipulating images
- OpenCV based algorithms whenever possible (e.g. point pair alignment)

## Max Functions
The main issue is that numpy does not provide a function that provides both the maximum and its coordinates in a data matrix. Its necessary to rearrange the matrix, find the maximum location and then convert it back to indices which then allow to obtain the maximum.
### 3D Max Function to find max in each plane
With CNN, often the max value of a score and also its location in the output matrix/tensor is needed in order to obtain the coordinates of the optimal object or bounding box.
```
out2D       = out3D.reshape(out3D.shape[0],-1)      # converd n,m,o array to n,m*o
idx         = out2D.argmax(1)                       # find max location in m*o range for each n
max_y,max_x = np.unravel_index(idx,out3D.shape[1:]) # unravel the location to m,o coordinates
max_prob    = out2D[np.arange(len(idx)),idx]        # obtain max and max location
```
### 3D Max Function to find max along axis
With CNN, the location of the maximum is needed when anchors are used.
```
k           = np.argmax(out3D,axis=0)               # max class score location along n axis
n,o         = out3D.shape[1:]                       # 
I,J         = np.ogrid[:n,:o]                       # there is max at each location of m,o
class_score = out_3D[k, I, J]                       # max class score
```
## Requirements
* NCNN
* numpy
* opencv
* cython is suggested

## Examples & Models

| Implementation | Author | Website | Article | Image Size| Implementation | Extraction [ms]  | Pipeline [ms]  |
|---|-----|-----|-----|-----|----|----|-----|
| **Object Detection** |
| fastestdet | Xueha Ma | [FastestDet](https://github.com/dog-qiuqiu/FastestDet)  | [Artcile](https://zhuanlan.zhihu.com/p/536500269)| 352x352 | anchor-free | 8.5 | 11.4 |
| *yolo5s* | | [QEngineering](https://github.com/Qengineering/YoloV5-ncnn-Jetson-Nano) | | 640x640 | | TBD | TBD|
| yolo7-tiny | Xiang Wu | [Original](https://github.com/xiang-wuu/ncnn-android-yolov7) [QEngineering](https://github.com/Qengineering/YoloV7-ncnn-Jetson-Nano)| [Article](https://arxiv.org/pdf/2207.02696.pdf) |  640, base=64|  variable input size  |  40 | 51 |
| *yolo8* | | | | | | TBD | TBD |
| *yolox (nano)* | Megvii-BaseDetection & FeiGeChuanShu | [Original](https://github.com/Megvii-BaseDetection/YOLOX) [QEngineering](https://github.com/Qengineering/YoloX-ncnn-Jetson-Nano) [Android](https://github.com/FeiGeChuanShu/ncnn-android-yolox) |   |  416x416 | | |  
| yolox (tiny) | | [Original](https://github.com/Megvii-BaseDetection/YOLOX) [QEngineering](https://github.com/Qengineering/YoloX-ncnn-Jetson-Nano) [Android](https://github.com/FeiGeChuanShu/ncnn-android-yolox) |  | 416x416 | | TBD | TBD |
| *yolox (small)* | | [Original](https://github.com/Megvii-BaseDetection/YOLOX) [QEngineering](https://github.com/Qengineering/YoloX-ncnn-Jetson-Nano) [Android](https://github.com/FeiGeChuanShu/ncnn-android-yolox) |  |   640x640 | | TBD | TBD |
| **Hand Detector** |
| blaze (palm-lite/full) Hand model| Vidur Satija (blazepalm), FeiGeChuanShu (android) | [blazepalm](https://github.com/vidursatija/BlazePalm) [Android](https://github.com/FeiGeChuanShu/ncnn-Android-mediapipe_hand) | |   192x192 |  | 8.09(full) 7.1(light) | 8.9(full)|
| *nanodet (nanodet-hand) Hand model* | | [FeiGeChuanShu](https://github.com/FeiGeChuanShu/ncnn_nanodet_hand/tree/main/ncnn-android-nanodet) [QEngineering](https://github.com/Qengineering/Hand-Pose-ncnn-Raspberry-Pi-4)|  | | | |
| *yolox (yolox_hand_relu/swish) Hand model*| | [FeiGeChuanShu](https://github.com/FeiGeChuanShu/ncnn_nanodet_hand/tree/main/ncnn-yolox-hand) [QEngineering](https://github.com/Qengineering/Hand-Pose-ncnn-Raspberry-Pi-4)|  | | | |
| *pfld (handpose)* | | [FeiGeChuanShu](https://github.com/FeiGeChuanShu/ncnn_nanodet_hand) [QEngineering](https://github.com/Qengineering/Hand-Pose-ncnn-Raspberry-Pi-4)|  | | | |
| **Hand Skeleton** |
| mediapipe (hand-lite/full-op) Skeleton model | Vidur Satija (blazepalm), FeiGeChuanShu (android) | [blazepalm](https://github.com/vidursatija/BlazePalm) [Android](https://github.com/FeiGeChuanShu/ncnn-Android-mediapipe_hand) | |   224x224 | 3D  | 6.8 (full) 5.1(light) | 7.1 (full) |
| **Face Detector** |
| *blazeface*  | | | |  |    |  |
| retinaface retinaface-R50 | Tencent | [Tencent](https://github.com/Tencent/ncnn/blob/master/python/ncnn/model_zoo/retinaface.py) [QEngineering](https://github.com/Qengineering/Face-Recognition-Raspberry-Pi-64-bits) | [Origin](https://github.com/deepinsight/insightface/) | variable, 320x320 | no scaling or padding needed | 3.6 | 7 |
| *scrfd*      | | | |  |    |  |
| *ultraface*  | | | |  |    |  |
| **Face Detector Support** |
| *live*  | yuan hao / Minivision_AI | [Website](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing-APK/) |  | 80x80 | dual stage | 3.5 | 5.3 |
| *mask*  | | |  | |    |  |
| **Face Recognition** |
| arface mobilefacenet | Xinghao Chen | [QEngineering](https://github.com/Qengineering/Face-Recognition-Raspberry-Pi-64-bits/tree/main) | [Origin](https://github.com/deepinsight/insightface) | 112x112 | no NMS, anchorfree  |  6  | 6.4  |
| **Person Detector** |
| ultralightpose, Person | Xueha Ma | [MobileNetV2-YOLOv3-Nano](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3) |  | 320x320  | no NMS, anchorfree   | 8.2 | 9.4 |
| blazeperson - [x] | Valentin Bazarevsky / Google | [Android Blazepose](https://github.com/FeiGeChuanShu/ncnn_Android_BlazePose) [Google](https://google.github.io/mediapipe/solutions/pose)| [Paper](https://arxiv.org/pdf/2006.10204.pdf)| 224x224 | | 9.4 | 11 |
| **Person Pose** |
| ultralightpose, Skeleton| Xueha Ma | [UltralightPose](https://github.com/dog-qiuqiu/Ultralight-SimplePose) |  | 192x256 | no NMS, anchorfree   | 5.84 |  6.4 |
| blazepose - [x] | Valentin Bazarevsky / Google | [Android Blazepose](https://github.com/FeiGeChuanShu/ncnn_Android_BlazePose) [DepthAI Blazepose](https://github.com/geaxgx/depthai_blazepose) | [Paper](https://paperswithcode.com/paper/blazepose-on-device-real-time-body-pose) | 256x256 | 3D | 6.3ms(light), 7.9ms(full), 22ms(heavy) | 6.6ms (light) |

## Implementation Notes
| Name       | Anchors  | Applciation of Anchors | NMS | Softmax | Sigmoid, Tanh |
|------------|----|---|---|---|---|
| age        | not implemented | | | | |
| race       | not implemented | | | | |
| arcface    | no anchors | none | no | none | none
| blazeface  | detectstride8,16 anchors | gnerate_proposals | NMS | none | sigmoid 
| blazehand hand | anchors 8,16,16,16 | none | NMS | warp | sigmoid
| blazehand skelet | uses mediapipe hand
| blazepose person | anchors 8,16,32,32,32 | decode_boxes | NMS | warp | sigmoid
| blazepose skelet  | no anchors | none | no | unwarp | none
| blur       | computes high frequency content in image
| fastestdet | no anchors | none | NSM | none |sigmoid, tanh
| handpose hand   | detectstride8,16,32 anchors | generate_proposals | NMS | softmax | none
| handpose skelet |  no anchors | none | no | none | none
| live       | 2 models | average of confidence of both models | | | | 
| mask       | not implemented yet | | | | |
| mediapipehandpose skeleton | no anchors | none | none | none | none
| retinaface | anchors, 8,16,32 | generateproposal | NMS | none | exp
| scrfd, 9 different models  | anchors 8,16,32             | genreateproposal | NMS  | none    | none
| ultraface  | anchors 8,16,32,64       | generateBBOX     | NMS  | none    | none
| ultralightpose person| no anchors        | none             | no | none    | none
| ultralightpose skelet| no anchors        | none             | no | none    | none 
| yolo5      | not implemented yet
| yolo7      | detectstride8,16,32 anchors | detect_stride | NMS  |         | sigmoid
| yolo8      | grid_strides table          | generate_proposal | NMS  | softmax | sigmoid
| yolox      | not implemented yet


## Affine and Warp Transformations
Images can be affine transformed or warped.
Affine transformation inlucdes scaling, rotation, translation and shear. The transformation uses a 2x3 Matrix and preserves the parallelism of lines.
 
        trans_mat = cv2.getAffineTransform(srcPts, dstPts)
        img  = cv2.warpAffine(src=img, M=trans_mat)
        src_pt = np.array([[(x, y)]], dtype=np.float32)
        dst_pt = cv2.transform(src_pt,trans_mat)         

Warped transformation includes perspective and distortion.
    
Affine transformation (scaling,rotation,translation,shear)

    R = [ [a11, a12, t13]
          [a21, a22, t23]
          [  0,   0,   1] ]
    
    dst = [ A | t ] {src}
          [ 0 | 1 ] {1} 
    
    dst = A * src + t
            
    dst_x = a11*x + a12*y + t13 
    dst_y = a21*x + a22*y + t23

Rotation around the origin

    R = [ [cos(θ), -sin(θ), 0]
          [sin(θ),  cos(θ), 0]
          [0     ,  0     , 1] ]

Rotation around center cx,cy, this is the same as T(cx,cy)∗R(θ)∗T(−cx,−cy)

    R = [ [cos(θ), -sin(θ), -cx⋅cos(θ)+cy⋅sin(θ)+cx]
          [sin(θ),  cos(θ), −cx⋅sin(θ)−cy⋅cos(θ)+cy]
          [0     ,  0     ,  1] ]

Translation

    T = [ [1, 0, dx],
          [0, 1, dy],
          [0, 0, 1]]

Scaling

    S = [ [sx, 0, 0],
          [0, sy, 0],
          [0,  0, 1]]

Shear


    H = [ [ 0, sx, 0],
          [sy,  0, 0],
          [ 0,  0, 1]]


There is also reflection and mirror image transformation.
    
Warp transformation

    R = [ [a11, a12, a13]
          [a21, a22, a23]
          [a31, a32, a33] ]

    dst = [ a11, a12, a13 ] {src}
          [ a21, a22, a23 ] {1} 
          ----------------------
          [ a31, a32 ,a33] {src}
                           {1}

## Installation

**ncnn** ```pip install ncnn ```

**camera** ```pip install camera-util``` 

**opencv** ```pip install opencv-contrib-python```  


### To install OpenCV on Raspi:  

5. ```cd ~```
6. ```sudo pip3 install opencv-contrib-python==4.5.3.56``` as time progresses the version numnber might need to be increased, but many newer version have installation issues.

## Example Programs

### blazeperson
This cnn extracts a bound box and 4 keypoints. The bounding box includes upper torso and head and they keypoints are the lower and upper bound of the torso, the forehead and somewhere above the head. It bounding box does not scale well with distance and torso keypoints imply often a rotation of up to 10 degrees that is not present.

**Examples**: 
* ```test_display.py``` testing of opencv display framerate, no camera, just refresh rate.

## Pip upload
Note to myself:
```
py -3 setup.py check
py -3 setup.py sdist
py -3 setup.py bdist_wheel
pip3 install dist/thenewpackage.whl
twine upload dist/*
```
## Changes
```
2023 - Initial Release
Urs Utzinger
```

## Documentation
### *utils.py*

Ideally all functions and objects in this code should be changed to adhere to opencv point, vector and rectangle structures.

    srcPts = np.array( [
        [ [point1.x, poin1.y] ],    
        [ [point2.x, point2.y] ], 
        [ [point3.x, point3.y] ], 
        [ [point4.x, point4.y] ]
    ], dtype=np.float32)

    matrix = np.array( [
        [scale, 0, l],
        [0, scale, t],
        [0,     0, 1]
    ], dtype=np.float32)

Utility functions
- clip(x,y) clip to 0..y
- clamp(x,smallest,largest)
- sigmoid_np(x)
- tanh_np(x)
- sigmoid(x)
- tanh(x)

Point support
- Point(x,y,visible)
  - distance(to other point)
  - angle(to other point)

Vector support
- Vector(point_a,point_b)
  - lenght
  - dot(with other vector)
  - dotprod(with other vector)
  - angle
  - mult(with other vector)
  - div(with other vector)
  - sub(with other vector)
  - add(with other vector)

Rectangle support
- Rect(x0,y0,x1,y1)
  - width
  - height
  - center
  - area
  - intersection_area(with other rectangle)
  - draw
  - resize(scale,left_add,top_add)

Object support
- objectTypes are 'rect', 'yolo80', 'person', 'preson17', 'person32', 'hand', 'hand7', 'face', 'face5'
- Object(x0,y0,x1,y1,probability,label,type,keypoints_x, keypoints_y, keypoints_visibility)
  - hasKeypoints
  - resize(scale,left_add,top_add)
  - draw
  - drawRect
  - angle for face5, hand7, person17, person32
- drawObjects
- resizeObjectRects(objects,scale,left_add,top_add)

Rotated Object support
- RotatedObject(score, landmarks, rotation, type, centerx, centery, width, height, rotatedrectangle(Point*4), skeleton, skeleton_score)
  - resize
  - draw
- createRotatedObject(object,rotation,rotation_x,rotation_y, square, scale) ceates new bounding box
- align(src_poins, dst_points) Umeyama algorithem

Resize images, Region of Interest Extraction
- calculateBox, modifies boundingbox with shift, scale, sets new width and height keeping center
- computeImgSize creates new w,h based on targetsize and base
- resizeImg new_with, new_height, pad on/off resize while maintaining aspect ratio
- resizeImg2Targetsize resizes image to targetsize, multiple of base, keeps aspect ratio
- extractObjectROI extracts region of interest, applies rotation if keypoints available, simple option: takes angele of a few keypoints, complex option:  takes all keypoints to compute affine transformation, uses warpaffine to extract ROI from original image, resulting image is square and of targetsize

### *utils_cnn.py*
Non Maxima Supression, all implementations work when different object labels are present.
- nms (fast)
- nms_combination
- nms_weighted (slower)
- Zscore of embeddings
- CosineDistance between two sets of embeddings
- EuclideanDistance between two sets of embeddings
- l2_normalize emobeddings x/sqrt(sum(x*x))

### *setup.py*
instruction to cythonize subroutins.

## References
