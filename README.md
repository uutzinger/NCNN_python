# NCNN Python

- [NCNN Python](#ncnn-python)
  * [Overview](#overview)
  * [Requirements](#requirements)
    + [OpenCV on Raspi](#opencv-on-raspi)
  * [Optimizations](#optimizations)
  * [Max Functions](#max-functions)
    + [3D Max Function to find max and location in each plane](#3d-max-function-to-find-max-and-location-in-each-plane)
    + [3D Max Function to find max and location along axis](#3d-max-function-to-find-max-and-location-along-axis)
  * [Implementations](#implementations)
  * [Notes](#notes)
  * [Test Programs](#test-programs)
  * [History](#history)
  * [Documentation](#documentation)
    + [**utils_object.py**](#--utils-objectpy--)
      - [Object Types](#object-types)
      - [Object Structure](#object-structure)
      - [Object Methods](#object-methods)
      - [LandMarksSmoothingFilter](#landmarkssmoothingfilter)
        * [OneEuroFilter](#oneeurofilter)
        * [LowPassFilter](#lowpassfilter)
    + [**utils_image.py**](#--utils-imagepy--)
    + [**utils_hand.py**](#--utils-handpy--)
    + [**utils_face.py**](#--utils-facepy--)
    + [**utils_cnn.py**](#--utils-cnnpy--)
    + [**utils_blaze.py**](#--utils-blazepy--)
    + [**utils_affine.py**](#--utils-affinepy--)
    + [**setup.py**](#--setuppy--)
  * [Affine and Warp Transformations](#affine-and-warp-transformations)
    + [Affine transformation (scaling,rotation,translation,shear)](#affine-transformation--scaling-rotation-translation-shear-)
    + [Warp transformation](#warp-transformation)
  * [Pip upload](#pip-upload)
  * [References](#references)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

## Overview
This is a collection of python programs using the NCNN framework. 
NCNN is an open source implementation of Convolutional Neural Networks by [TENCENT](https://en.wikipedia.org/wiki/Tencent). NCNN is optimized for mobile platforms; often it is faster to run the model on CPU than GPU.

Code was converted to Python and accelerated with numpy and cython. A basic framework to handle models, objects, bounding boxes, keypoints and transformations was developed. Python versions for Non Maximum Suppression and image manipulation was implemented. In general developing code in python does not accelerate existing C programs. The motivation for this work was to provide optimized examples for the Python platform.

There are several sites listing current implementations of CNN models that have been converted to NCNN. However, there is no single authoritative repository:
* [Tencent](https://github.com/Tencent/ncnn/blob/master/python/ncnn/model_zoo)
* [Baiyuetribe](https://github.com/Baiyuetribe/ncnn-models)
* [Marton Juhasz](https://github.com/nilseuropa/ncnn_models)

## Requirements
To run the program you will need to install the following packages:
* NCNN
* numpy
* opencv
* cython is suggested

**ncnn** ```pip install ncnn ```

**camera** ```pip install camera-util``` 

**opencv** ```pip install opencv-contrib-python```  

### OpenCV on Raspi  

```sudo pip3 install opencv-contrib-python==4.5.3.56``` 

As time progresses the version number might need to be increased, but many newer version have installation issues.

## Optimizations

To increase performance, the following rules were observed:
- Use OpenCV based algorithms whenever possible
- OpenCV data manipulations are faster than Numpy
- OpenCV is slightly faster than NCCN when manipulating images (e.g. resize)
- Avoid indexing over tensor/matrix dimensions
- Use Numpy boolean indexing when possible, avoid ```np.where```
- ```np.append```, ```np.vstack``` or ```np.stack``` should be avoided when adding small amounts of data and regular lists and append function can be used
- If ```np.concatenate``` is needed, apply it once to a list of all objects that need concatenation
- If ```np.max``` and ```np.argmax``` are needed for a 3D array use examples shown below

Torch dependencies were removed. A useful guide is listed here: [torch to numpy](https://medium.com/axinc-ai/conversion-between-torch-and-numpy-operators-ce189b3882b1)

## Max Functions
Numpy does not provide a function that provides both the maximum and its indices in a data matrix. Its necessary to rearrange the matrix, find the maximum location and then convert it back to indices. Often maximum is needed to threshold and indices are needed for further location calculations.
### 3D Max Function to find max and location in each plane
Often needed to threshold score and find keypoints or bounding boxes.
```
out2D       = out3D.reshape(out3D.shape[0],-1)      # convert n,m,o array to n,m*o
idx         = out2D.argmax(1)                       # find max location in m*o range for each n
max_y,max_x = np.unravel_index(idx,out3D.shape[1:]) # unravel the location to m,o coordinates
max_out     = out2D[np.arange(len(idx)),idx]        # obtain max
```
### 3D Max Function to find max and location along axis
The location of the maxima along axis is needed when anchors are used in object detection.
```
k           = np.argmax(out3D,axis=0)               # max class score location along n axis
n,o         = out3D.shape[1:]                       # 
I,J         = np.ogrid[:n,:o]                       # there is max at each location of m,o
class_score = out_3D[k, I, J]                       # max class score
```

## Implementations

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
| arcface mobilefacenet | Xinghao Chen | [QEngineering](https://github.com/Qengineering/Face-Recognition-Raspberry-Pi-64-bits/tree/main) | [Origin](https://github.com/deepinsight/insightface) | 112x112 | no NMS, anchorfree  |  6  | 6.4  |
| **Person Detector** |
| ultralightpose, Person | Xueha Ma | [MobileNetV2-YOLOv3-Nano](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3) |  | 320x320  | no NMS, anchorfree   | 8.2 | 9.4 |
| blazeperson - [x] | Valentin Bazarevsky / Google | [Android Blazepose](https://github.com/FeiGeChuanShu/ncnn_Android_BlazePose) [Google](https://google.github.io/mediapipe/solutions/pose)| [Paper](https://arxiv.org/pdf/2006.10204.pdf)| 224x224 | | 9.4 | 11 |
| **Person Pose** |
| ultralightpose, Skeleton| Xueha Ma | [UltralightPose](https://github.com/dog-qiuqiu/Ultralight-SimplePose) |  | 192x256 | no NMS, anchorfree   | 5.84 |  6.4 |
| blazepose - [x] | Valentin Bazarevsky / Google | [Android Blazepose](https://github.com/FeiGeChuanShu/ncnn_Android_BlazePose) [DepthAI Blazepose](https://github.com/geaxgx/depthai_blazepose) | [Paper](https://paperswithcode.com/paper/blazepose-on-device-real-time-body-pose) | 256x256 | 3D | 6.3ms(light), 7.9ms(full), 22ms(heavy) | 6.6ms (light) |

## Notes
| Name       | Anchors  | Application of Anchors | NMS | Softmax | Sigmoid, Tanh |
|------------|----|---|---|---|---|
| age        | not available | | | | |
| race       | not available | | | | |
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

## Test Programs
- test_arcface: runs retinaface to find faces then extracts ROI and applies arcface to compute embeddings
- test_blazehandpose: uses blaze palm to find palms then runs handpose to find sceleton
- test_balzeperson: finds people
- test_blazepersonpose: finds people then runs skeleton detection
- test_blur: extracts ROI and assess if image is blurred
- test_fastestdet
- test_gestures: detects palm then extracts ROI, calculates sceleton then inteprets hand sign
- test_handpose: finds palms then computes skeleton
- test_live: determines if image of face is live or fake
- test_retinaface: detects faces
- test_yolo7: detects objects

## History
```
2023 - Initial Release
Urs Utzinger
```

## Documentation
Documentation of utility functions:

### **utils_object.py**

#### Object Types
```
objectTypes = {'rect':0, 'yolo80':1, 'hand':2, 'palm7':3, 'hand21':4, 'face':5, 
               'face5':6, 'person':7, 'person4':8, 'person17':9, 'person39':10 }
```
Simple objects with bounding box: rect, hand, face, person

Object with keypoints, plam7, hand21, face5, person4, person17, person39, where number indicates the number of keypoints.

Ojects: yolo80 (80 classes)

#### Object Structure
```
object.type=objectTypes['rect'],  # Object Type
object.bb = np.array( [           # Bounding Box, 4 or 2 points
                      [ [-1, -1] ],
                      [ [-1, -1] ],
                      [ [-1, -1] ],
                      [ [-1, -1] ]
                      ], dtype=np.float32 )
object.p  = -1.                   # Probability
object.l  = -1                    # Label number
object.k  = []                    # Keypoints 
object.v  = []                    # Keypoints visibility 
```
#### Object Methods
True or False:
- hasKeypoints, hasVisibility, isRotated, is1D, is2D, is3D

Regular:
- extent: max-min of bounding box
- center: center of bounding box
- width_height: on rotated bounding box
- relative2absolute: scale from 0..1 to 0.. width/height
- transform: apply cv2.transfrom to bounding box and keypoints
- intransform: inverse transform 
- resize: resize and shift bounding box
- square: ensure square bounding box, takes largest dimension
- angle: angle of keypoints face5, palm7, person4, person17, person 39
- rotateBoundingBox: rotate rectangualr bounding box by angle
- draw: draw the bounding boxe and keypoints 
- drawRect: draw bounding box 
- printText: prints text to top left corner of bounding box

- drawObjects: draw multiple objects
- calculateBox: phased out

#### LandMarksSmoothingFilter
- get_object_scale
- apply
- get_alpha
- reset

##### OneEuroFilter
- get_alpha
- apply

##### LowPassFilter
- apply
- apply_with_alpha
- has_last_raw_value
- last_raw_value
- last_value
- reset

### **utils_image.py**
- resizeImage to new width and new height, padding optional
- resizeImage2TargetSize so that width or height is targetsize and pad so that width or height is multiple of base
- extractObjectROI extract image from any bounding box and scale to targetsize
- extractRectROI, extract image from un-roated bounding box

### **utils_hand.py**
- gesture of handsceleton will select none, point, swear, thumbs up,down,left,right, vulcan, oath, paper, rock, victory, finger, hook, pinky, one, two, three, four, ok

### **utils_face.py**
- overlaymatch places found face on top of face

### **utils_cnn.py**
- nms_cv, non maximum supppresion, filters bounding boxes based on overlap, uses openCV nms, fastest approach
- nms, simple, fully python based
- nms_combination, likely not needed
- nms_weighted, uses weighted approach for overlapping bounding boxes
- nms_blaze original python code for blaze, includes mathematical explanations

- matchEmbeddings, given one embedding and comparing it to a list of embedings, finds the one closest matching
- Zscore (data - mean(data)) / std(data)
- CosineDistance between two embeddings
- EuclidianDistnace between two embeddings
- l2normalize x/sqrt(sum(x*x)), try cv2.norm instead
- findThreshold, might be obsolete

### **utils_blaze.py**
- decode_boxes
- Anchor object
- Anchor Params
- generate_anchors
- generate_anchors_np

### **utils_affine.py**
- composeAffine (T,R,Z,S) creates transformation matrix from translation, rotation, zoom/scale, shear
- decomposeAffine23 converts transformation matrix back to T,R,Z,S
- decomposeAffine44 converst transfomration matrix to T,R,Z,S 

### **setup.py**
Instruction to cythonize subroutines

## Affine and Warp Transformations
Region of interested can be extracted from images using affine or warped transformation.

- Affine transformation inlucdes scaling, rotation, translation and shear. The transformation uses a 2x3 Matrix and preserves the parallelism of lines.

- Warped transformation includes perspective and distortion.

### Affine transformation (scaling,rotation,translation,shear)

    trans_mat = cv2.getAffineTransform(srcPts, dstPts)
    img  = cv2.warpAffine(src=img, M=trans_mat)
    src_pt = np.array([[(x, y)]], dtype=np.float32)
    dst_pt = cv2.transform(src_pt,trans_mat)         


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
    
### Warp transformation

    R = [ [a11, a12, a13]
          [a21, a22, a23]
          [a31, a32, a33] ]

    dst = [ a11, a12, a13 ] {src}
          [ a21, a22, a23 ] {1} 
          ----------------------
          [ a31, a32 ,a33] {src}
                           {1}

## Pip upload
Note to myself:
```
py -3 setup.py check
py -3 setup.py sdist
py -3 setup.py bdist_wheel
pip3 install dist/thenewpackage.whl
twine upload dist/*
```

## References
