###########################################################
# NCNN utility data types and functions
# Urs Utzinger, Spring 2022
###########################################################

import numpy as np
import cv2
import cython
from math import floor, sqrt, atan2, pi, exp, degrees, isclose
from copy import deepcopy

@cython.cfunc
def clip(x: cython.double, y: cython.double) -> cython.double:
    ''' 
    Clip x to [0,y]
    '''
    return 0  if x < 0 else y if x > y else x 

# Numpy has its own clip function

@cython.cfunc
def clamp(val: cython.double, smallest: cython.double, largest: cython.double): 
    '''
    Clip val to [smallest, largest]
    '''
    if val < smallest: return smallest
    if val > largest: return largest
    return val

@cython.cfunc
def sigmoid_np(x):
    return 1. / (1. + np.exp(-x))

@cython.cfunc
def sigmoid(x):
    return 1. / (1. + exp(-x))

@cython.cfunc
def tanh(x):
    return 2.0 / (1.0 + exp(-2.*x)) - 1

@cython.cfunc
def tanh_np(x):
    return 2.0 / (1.0 + np.exp(-2.*x)) - 1

########################################################################################
# Object
########################################################################################

objectTypes = {'rect':0, 'yolo80':1, 'hand':2, 'palm7':3, 'hand21':4, 'face':5, 
               'face5':6, 'person':7, 'person4':8, 'person17':9, 'person39':10 }
# objectX is object with X keypoints

class Object(object):
    '''
    Object consists of:
     - Object type
     - Bounding Box: 
        - two pointst (top left, bottom right) 
        - 4 points or (top left, top right, bottom right, bottom left)
     - OjectType: Rect, Yolo, Person, Hand, Face etc.
     - Probability of Object or Keypoints in Bounding Box
     - Label if Yolo Object
     - Keypoints:
        - single vector for embeddings
        - 2D vector for x/y coordinates
        - 3D vector for x/y/z coordinates    
     - Keypoint visibility
    In two stage aglorithms you would obtain 
     - a bounding box with landmarks (hand, face, person)
     - then a skeleton, embeddings or keypoints within the boundingbox
    resulting in two objects.
    
    2Dpoints: np.array([ [[x1, y1]],     ..., [[xn, yn]]     ])
    3Dpoints: np.array([ [[x1, y1, z1]], ..., [[xn, yn, zn]] ])
    
    '''
    def __init__(self, 
                 type=objectTypes['rect'], # Object Type
                 bb = np.array( [       # Bounding Box
                    [ [-1, -1] ],
                    [ [-1, -1] ],
                    [ [-1, -1] ],
                    [ [-1, -1] ]
                 ], dtype=np.float32),  
                 p  = -1.,              # Probability
                 l  = -1,               # Label number
                 k  = [],               # Keypoints 
                 v  = []):              # Keypoints visibility (Ture/False)
        self.type  = type # Object Type
        self.bb = bb      # Bounding Box
        self.p  = p       # Probability of Object in Bounding Box
        self.l  = l       # Label, object number if applicable.
        self.k  = k       # Key Points
        self.v  = v       # Key Points visibility (True/False)

        self.color0 = (100, 100, 100) # grey
        self.color1 = (255, 255, 255) # white
        self.color2 = ( 55, 255,  55) # green  
        self.color3 = ( 55,  55, 255) # blue  
        self.color4 = (255,  55,  55) # red  
        self.color5 = (255, 138,   0) # blueish
        self.color6 = (  0, 255, 255) # cyan
        self.color7 = (  0,   0, 255) # blue
        self.color8 = (255, 255,   0) # yellow
        
    def hasKeypoints(self):
        """ Kyepoints are populated """
        return True if (len(self.k) > 0) else False

    def hasVisibility(self):
        """ Visibility for keypoints is populated """
        return True if (len(self.v) > 0) else False

    def isRotated(self):
        """ Bounding Box is rotated as it has 4 corner points """
        return True if (self.bb.shape[0] > 2) else False

    def is1D(self):
        """ Keypoints are one dimensional """
        return True if (self.k.shape[2] == 1) else False

    def is2D(self):
        """ Keypoints are two dimensional """
        return True if (self.k.shape[2]  == 2) else False

    def is3D(self):
        """ Keypoints are three dimensional """
        return True if (self.k.shape[2] == 3) else False

    def extent(self):
        """
        Computes width and height of bounding box
        """
        return (np.max(self.bb,axis=0, keepdims=True) - np.min(self.bb,axis=0, keepdims=True))

    def center(self):
        """
        Computes center of bounding box which is the averge of all points
        """
        return np.mean(self.bb, axis=0, keepdims=True)

    def width_height(self):    
        if self.isRotated():
            # Bounding Box is: top left, top right, bottom right, bottom left
            w = (cv2.norm(self.bb[1,:,:] - self.bb[0,:,:])) # top right - top left
            h = (cv2.norm(self.bb[2,:,:] - self.bb[1,:,:])) # bottom right - top right
        else:
            # Bounding Box is: top left, bottom right
            w = (self.bb[1,0,0] - self.bb[0,0,0])
            h = (self.bb[1,0,1] - self.bb[0,0,1])
        return (w, h)
    
    def relative2absolute(self, height:cython.double, width:cython.double):
        """
        Adjust bounding box and keypoints from relative cooridinates to absolute.
        Relative is on scale from 0..1
        Absolute is from 0 to width or height
        """ 
        self.bb[:,0,0]  *= width
        self.bb[:,0,1]  *= height
        if self.hasKeypoints():
            self.k[:,0,0]  *= width 
            self.k[:,0,1]  *= height

    def transform(self, trans_mat):
        """
        Use affine transformation on bounding box and keypoints
        If 3D points provided only apply in x-y plane
        """
        self.bb = cv2.transform(self.bb, m=trans_mat)
        if self.is2D():
            self.k = cv2.transform(self.k, m=trans_mat)        
        elif self.is3D():
            self.k[:,:,0:2] = cv2.transform(self.k[:,:,0:2], m=trans_mat)
            # estimate scaling if no shear and proportional scaling
            if isclose(trans_mat[1,0], -trans_mat[0,1], abs_tol=1e-5):
                s  = sqrt(  trans_mat[0,0]*trans_mat[0,0] + trans_mat[1,0]*trans_mat[1,0] )
                self.k[:,0,2] *= s
                
    def invtransform(self, trans_mat):
        """
        Use inverse affine transformation on bounding box and keypoints
        """
        self.transform(cv2.invertAffineTransform(trans_mat))
    
    def resize(self, scale:cython.double, l:cython.int, t:cython.int, d=0, bounds = (-1.,-1.,-1.)):
        """
        Resized and shifts rectangles and landmarks in a list of objects.
        Shift l to left, t from top, then scale down by scale.
        If bounds (height, width, depth) are giving will limit location to 0..bound.
        This does NOT allocate new object in memory.
        """
        # bounding box
        self.bb[:,0,0]  = (self.bb[:,0,0] - l)/scale
        self.bb[:,0,1]  = (self.bb[:,0,1] - t)/scale
        if bounds[1] > 0:
            self.bb[:,0,0] = np.clip(self.bb[:,0,0], a_min=0, a_max=bounds[1])
        if bounds[0] > 0:
            self.bb[:,0,1] = np.clip(self.bb[:,0,1], a_min=0, a_max=bounds[1])

        # keypoints
        if self.is2D():
            self.k[:,0,0]  = (self.k[:,0,0] - l)/scale
            self.k[:,0,1]  = (self.k[:,0,1] - t)/scale
            if bounds[1] > 0:
                self.bb[:,0,0] = np.clip(self.bb[:,0,0], a_min=0, a_max=bounds[1])
            if bounds[0] > 0:
                self.bb[:,0,1] = np.clip(self.bb[:,0,1], a_min=0, a_max=bounds[0])
            
        elif self.is3D():
            self.k[:,0,0]  = (self.k[:,0,0] - l)/scale
            self.k[:,0,1]  = (self.k[:,0,1] - t)/scale
            self.k[:,0,2]  = (self.k[:,0,2] - d)/scale
            if bounds[0] > 0:
                self.k[:,0,1] = np.clip(self.k[:,0,1], a_min=0, a_max=bounds[0])
            if bounds[1] > 0:
                self.k[:,0,0] = np.clip(self.k[:,0,0], a_min=0, a_max=bounds[1])
            if bounds[2] > 0:
                self.k[:,0,2] = np.clip(self.k[:,0,2], a_min=0, a_max=bounds[2])

    def square(self, scale=1.0):
        ''' 
        creates square bounding box from rectangular bounding box
        '''
        if self.isRotated():
            # Bounding Box is: top left, top right, bottom right, bottom left
            w2 = (cv2.norm(self.bb[1,:,:] - self.bb[0,:,:]))/2.*scale # top right - top left
            h2 = (cv2.norm(self.bb[2,:,:] - self.bb[1,:,:]))/2.*scale # bottom right - top right
            if w2 > h2: h2 = w2
            else:       w2 = h2
            center = np.mean(self.bb, axis=0)
            l = (self.bb[3] - self.bb[0])[0] # left side
            angle = pi/2 - atan2(l[1],l[0])  # target angle is 90deg (vertical)
            ptl = center + np.array([[-w2, -h2]], dtype=np.float32)
            ptr = center + np.array([[ w2, -h2]], dtype=np.float32)
            pbr = center + np.array([[ w2,  h2]], dtype=np.float32)
            pbl = center + np.array([[-w2,  h2]], dtype=np.float32)
            srcPts = np.array( [ ptl, ptr, pbr, pbl ], dtype=np.float32)
            trans_mat = cv2.getRotationMatrix2D(center=(center[0,0],center[0,1]), angle=angle*180./pi, scale=1.)
            self.bb = cv2.transform(srcPts,trans_mat)
        else:
            # Bounding Box is: top left, bottom right
            w2 = (self.bb[1,0,0] - self.bb[0,0,0])/2.*scale
            h2 = (self.bb[1,0,1] - self.bb[0,0,1])/2.*scale
            center = np.mean(self.bb, axis=0)[0]
            if w2 > h2: h2 = w2
            else:       w2 = h2
            self.bb = np.array( [
                    [[ center[0] - w2, center[1] - h2 ]],
                    [[ center[0] + w2, center[1] + h2 ]] 
                    ], dtype=np.float32)
            
    def angle(self):
        if self.type == objectTypes['face5']:
            # Traget angle is 0 radians
            # Desired inverse rotation is counter clockwise
            # Eye
            target_angle = 0
            dy = (self.k[1,0,1] - self.k[0,0,1])
            dx = (self.k[1,0,0] - self.k[0,0,0])
            dl = sqrt(dx*dx + dy*dy) # length of vector
            dy_eye=dy/dl # normalize
            dx_eye=dx/dl            
            # Mouth
            dy = (self.k[4,0,1] - self.k[3,0,1])
            dx = (self.k[4,0,0] - self.k[3,0,0])
            dl = sqrt(dx*dx + dy*dy)
            dy_mouth=dy/dl
            dx_mouth=dx/dl
            # Average angle: we can not average the angle directly because 
            # of the discontinuity at 180 degrees but we can average the normalized vectors
            dx=dx_eye+dx_mouth
            dy=dy_eye+dy_mouth
            # rotation = target_angle - atan2(-dy,dx)
            # rotation = rotation - 2*pi * floor((rotation + pi)/2/pi)
            # return rotation # -pi .. +pi
            return atan2(dy,dx)
        
        elif self.type == objectTypes['palm7']:
            target_angle = pi/2
            dy = self.k[2,0,1] - self.k[0,0,1]
            dx = self.k[2,0,0] - self.k[0,0,0]
            rotation = target_angle - atan2(-dy, dx)
            # a - 2pi * floor((a + pi)/(2pi)) is a way to mapangle to the range [-pi,pi]
            rotation = rotation - 2*pi * floor((rotation + pi)/2/pi)            
            return rotation
        
        elif self.type == objectTypes['person4']:
            # has 4 landmarks
            # https://google.github.io/mediapipe/solutions/pose
            # https://en.wikipedia.org/wiki/Vitruvian_Man#/media/File:Da_Vinci_Vitruve_Luc_Viatour.jpg
            # Key point 0 - mid hip center
            # Key point 1 - point that encodes size & rotation (for full body)
            # Key point 2 - mid shoulder center
            # Key point 3 - point that encodes size & rotation (for upper body)
            #
            # Body center is landmark 0
            # Body width is 2 * landmark 0,1
            #
            dx = (self.k[1,0,0] - self.k[0,0,0])
            dy = (self.k[1,0,1] - self.k[0,0,1])
            target_angle = pi/2 # 90 degrees up
            rotation = target_angle - atan2(-dy, dx)
            # a - 2pi * floor((a + pi)/(2pi)) is a way to mapangle to the range [-pi,pi]
            rotation = rotation - 2*pi * floor((rotation + pi)/2/pi)                
            return rotation

        elif self.type == objectTypes['person17']:
            # Spine angle ultralightpose
            # {0, 1}, {1, 3}, {0, 2}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}

            # Need figure out which keypoints are the spine
            # https://github.com/tensorflow/tfjs-models/tree/master/pose-detection
            
            target_angle = pi/2
            # Left Torso 11-5
            dy = (self.k[5,0,1] - self.k[11,0,1]) # left hip to left shoulder
            dx = (self.k[5,0,0] - self.k[11,0,0])
            dl = sqrt(dx*dx + dy*dy) # length of vector
            dy_l=dy/dl # normalize
            dx_l=dx/dl            
            # Right Torso 12-6
            dy = (self.k[6,0,1] - self.k[12,0,1]) # right hip to right shoulder
            dx = (self.k[6,0,0] - self.k[12,0,0])
            dl = sqrt(dx*dx + dy*dy)
            dy_r=dy/dl
            dx_r=dx/dl
            # Average angle: we can not average the angle directly because 
            # of the discontinuity at 180 degrees but we can average the normalized vectors
            dx=dx_l+dx_r
            dy=dy_l+dy_r
            rotation = target_angle - atan2(-dy, dx)
            # a - 2pi * floor((a + pi)/(2pi)) is a way to mapangle to the range [-pi,pi]
            rotation = rotation - 2*pi * floor((rotation + pi)/2/pi)            
            return rotation # -pi .. +pi

        elif self.type == objectTypes['person39']:
            # Spine angle
            # https://developers.google.com/ml-kit/vision/pose-detection
            # https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
            # 23 (hip r) to 11 (sholder r) and 24 (hip l) to 12 (sholder l) are the torso
            target_angle = pi/2
            # Left Torso
            dy = (self.keypoints_y[11] - self.keypoints_y[23]) # left hip to left shoulder
            dx = (self.keypoints_x[11] - self.keypoints_x[23])
            dl = sqrt(dx*dx + dy*dy) # length of vector
            dy_l=dy/dl # normalize
            dx_l=dx/dl            
            # Right Torso
            dy = (self.keypoints_y[12] - self.keypoints_y[24]) # right hip to right shoulder
            dx = (self.keypoints_x[12] - self.keypoints_x[24])
            dl = sqrt(dx*dx + dy*dy)
            dy_r=dy/dl
            dx_r=dx/dl
            # Average angle: we can not average the angle directly because 
            # of the discontinuity at 180 degrees but we can average the normalized vectors
            dx=dx_l+dx_r
            dy=dy_l+dy_r
            rotation = target_angle - atan2(-dy, dx)
            # a - 2pi * floor((a + pi)/(2pi)) is a way to mapangle to the range [-pi,pi]
            rotation = rotation - 2*pi * floor((rotation + pi)/2/pi)            
            return rotation # -pi .. +pi

        else:
            return 0.

    def rotateBoundingBox(self):
        if self.isRotated():
            return
        else:
            # create rotated bounding box
            rotation = self.angle()
            center   = self.center()[0]
            trans_mat_rot = cv2.getRotationMatrix2D(center=center[0], angle=degrees(-rotation), scale=1.)
            # new bounding box
            ptl = self.bb[0,:,:]                               # top left
            ptr = np.array([[self.bb[1,0,0], self.bb[0,0,1]]]) # top right
            pbr = self.bb[1,:,:]                               # bottom right
            pbl = np.array([[self.bb[0,0,0], self.bb[1,0,1]]]) # bottom left
            bb = np.array( [ptl, ptr, pbr, pbl], dtype=np.float32)            
            self.bb = cv2.transform(bb, m=trans_mat_rot)

    # def rotateBoundingBox(self, rotation:cython.double, r_x:cython.double, r_y:cython.double, square=True, scale=1.0):
    #     """
    #     Convert 2 point bounding box to rotatedf 4 point bounding box
    #     Input: rotation and rotation center as fraction of width and height
    #     Bounding box will become square if enabled
        
    #     Rotation around center cx,cy, this is T(cx,cy)∗R(θ)∗T(−cx,−cy)
    #     R = [ [cos(θ), -sin(θ),  cx⋅cos(θ)+cy⋅sin(θ)+cx]
    #         [sin(θ),  cos(θ), −cx⋅sin(θ)−cy⋅cos(θ)+cy]
    #         [0     ,  0     ,  1] ]
    #     """

    #     c = self.center()
    #     cx = c[0,0,0]
    #     cy = c[0,0,1]

    #     if r_x == 0.0 and r_y == 0.0:
            
    #         trans_mat_rot = cv2.getRotationMatrix2D(center=(cx, cy), angle=degrees(-rotation), scale=scale)
    #         # convert self.bb to 4 point bounding box
    #         srcPts = np.array( [
    #             [ [self.bb[0,0,0], self.bb[0,0,1]] ], # top left
    #             [ [self.bb[1,0,0], self.bb[0,0,1]] ], # top right
    #             [ [self.bb[1,0,0], self.bb[1,0,1]] ], # bottom right
    #             [ [self.bb[0,0,0], self.bb[1,0,1]] ]  # bottom left
    #         ], dtype=np.float32)
    #         self.bb = cv2.transform(srcPts, m=trans_mat_rot)

    #     else:
    #         e = self.extent()
    #         w = e[0,0,0]
    #         h = e[0,0,1]
    #         if square: 
    #             if w > h: h = w
    #             else:     w = h
    #         # Compute rotated bounding box
    #         # Rotate the center of the bounding box around the center of rotation
    #         #  rx=0/ry=0   is the center of the bounding box
    #         #  rx=0/ry=0.5 is the bottom center of the bounding box
    #         if r_x == 0.0: shift_x = 0.0
    #         else:          shift_x = r_x * w
    #         if r_y == 0.0: shift_y = 0.0 
    #         else:          shift_y = r_y * h 

    #         trans_mat_rot = cv2.getRotationMatrix2D(center=(cx+shift_x, cy+shift_y), angle=degrees(-rotation), scale=scale)

    #         dx = w / 2. # half width
    #         dy = h / 2. # half height    
    #         srcPts = np.array( [
    #             [ [cx-dx, cy-dy] ], # top left
    #             [ [cx+dx, cy-dy] ], # top right
    #             [ [cx+dx, cy+dy] ], # bottom right
    #             [ [cx-dx, cy+dy] ]  # bottom left
    #         ], dtype=np.float32)

    #         self.bb = cv2.transform(srcPts, m=trans_mat_rot)
        
    def draw(self, img, color=(255, 255, 255), prob=True):
        """ 
        Draw object, rectangle, keypoints, label, probability
        """
        
        img_h, img_w = img.shape[:2]
                
        bb = np.int32(self.bb)
        if len(self.k) > 0:
            k  = np.int32(self.k)
        
        # probability
        if self.p > 0. and prob == True:
            text = "{:.1f}%".format(self.p * 100)
            (probLabel_width, probLabel_height), probBaseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            h = probLabel_height + probBaseLine
            w = probLabel_width
            x = bb[0,0,0]
            y = bb[0,0,1] - h
            if y                   < 0:     y = 0
            elif y                 > img_h: y = img_h - h
            if x + probLabel_width > img_w: x = img_w - w
            elif x                 < 0:     x = 0
            cv2.rectangle(img,
                (x, y),
                (x + w, y + h),
                self.color1, -1, )
            cv2.putText(img, text, (x, y + probLabel_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), )

        if self.type == objectTypes['rect'] or self.type == objectTypes['face']:
            # has no key points or labels
            pass
        
        elif self.type == objectTypes['yolo80']:
            # Yolo
            class_names = (
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush")
            # object label
            if self.label >= 0:
                text = "{}".format(class_names[int(self.label)])
                (objectLabel_width, objectLabel_height), objectBaseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                h = objectLabel_height + objectBaseLine
                x = bb[0,0,0]
                y = bb[-1,0,1] - h
                if y                   < 0:     y = 0
                elif y                 > img_h: y = img_h - h
                if x + objectLabel_width > img_w: x = img_w - objectLabel_width
                elif x                 < 0:     x = 0
                cv2.rectangle(img,
                    (x, y),
                    (x + objectLabel_width, y + h),
                    self.color1, -1, )
                cv2.putText(img, text, (x, y + objectLabel_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), )

        elif self.type == objectTypes['palm7']:
            # hand reference points          
            lk=k.shape[0]  
            for i in range(lk):
                text = "{}".format(i)
                (Label_width, Label_height), BaseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                h = Label_height + BaseLine
                x = k[i,0,0]
                y = k[i,0,1] - h
                cv2.rectangle(img,
                    (x, y),
                    (x + Label_width, y + h),
                    self.color1, -1, )
                cv2.putText(img, text, (x, y + Label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), )
                cv2.circle(img, (k[i,0,0],k[i,0,1]),4,self.color0,-1)

        elif self.type == objectTypes['hand21']:
            # https://medium.com/axinc-ai/blazehand-a-machine-learning-model-for-detecting-hand-key-points-c3943b82739a
            if len(k)>0:
                lk=k.shape[0]   
                for i in range(lk):
                    x0=k[i,0,0]
                    y0=k[i,0,1]
                    cv2.circle(img, (x0,y0),4, self.color4, -1)
                    if (i < lk-1):
                        x1=k[i+1,0,0]
                        y1=k[i+1,0,1]
                        if   (i <  4):            cv2.line(img, (x0,y0), (x1,y1), self.color2, 2, 8)
                        elif (i <  8 and i >  4): cv2.line(img, (x0,y0), (x1,y1), self.color2, 2, 8)
                        elif (i < 12 and i >  8): cv2.line(img, (x0,y0), (x1,y1), self.color2, 2, 8)
                        elif (i < 16 and i > 12): cv2.line(img, (x0,y0), (x1,y1), self.color2, 2, 8)
                        elif (i < 20 and i > 16): cv2.line(img, (x0,y0), (x1,y1), self.color2, 2, 8)

                x0=k[0,0,0]
                y0=k[0,0,1]
                x1=k[5,0,0]
                y1=k[5,0,1]
                cv2.line(img, (x0, y0), (x1,y1),  self.color3, 2, 8)
                x1=k[9,0,0]
                y1=k[9,0,1]
                cv2.line(img, (x0, y0), (x1,y1),  self.color3, 2, 8)
                x1=k[13,0,0]
                y1=k[13,0,1]
                cv2.line(img, (x0, y0), (x1,y1),  self.color3, 2, 8)
                x1=k[17,0,0]
                y1=k[17,0,1]
                cv2.line(img, (x0, y0), (x1,y1),  self.color3, 2, 8)
                
        elif self.type == objectTypes['face5']:
            # keypoints
            #  0 right eye
            #  1 left eye
            #  2 nose
            #  3 mouth right
            #  4 mouth left
            if ( self.hasKeypoints()):
                cv2.circle(img, (k[0,0,0], k[0,0,1]), 2, self.color6, -1)
                cv2.circle(img, (k[1,0,0], k[1,0,1]), 2, self.color6, -1)
                cv2.circle(img, (k[2,0,0], k[2,0,1]), 2, self.color7, -1)
                cv2.circle(img, (k[3,0,0], k[3,0,1]), 2, self.color8, -1)
                cv2.circle(img, (k[4,0,0], k[4,0,1]), 2, self.color8, -1)
           
        elif self.type == objectTypes['person4']:
            # Extended arms =  Height of man
            # Navel from bottom of feet (sqrt(5/4) - 1/2) * Height of man
            
            # Key point 0 - mid hip center
            # Key point 1 - point that encodes size & rotation (for full body) (top of head)
            # Key point 2 - mid shoulder center
            # Key point 3 - point that encodes size & rotation (for upper body) (vitruvian man circle goes through this point)
            # 
            # Height of man is 2 * Keypoint 0 to Keypoint 1
            # Bottom of man is extension of keypoint 0 to keypoint 1 by half of height height
                        
            # Person reference points  
            lk = k.shape[0]   
            for i in range(lk):
                text = "{}".format(i)
                (Label_width, Label_height), BaseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                h = Label_height + BaseLine
                x = int(k[i,0,0])
                y = int(k[i,0,1] - h)
                cv2.rectangle(img,
                    (x, y),
                    (x + Label_width, y + h),
                    self.color1, -1, )
                cv2.putText(img, text, (x, y + Label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), )
                cv2.circle(img, (k[i,0,0],k[i,0,1]),4,self.color0,-1)

            nc       = sqrt(5/4) - 0.5
            dx       = k[1,0,0] - k[0,0,0]
            dy       = k[1,0,1] - k[0,0,1]
            bottom_x = k[0,0,0] - dx
            bottom_y = k[0,0,1] - dy
            h        = 2.*sqrt(dx*dx + dy*dy)
            # Vitruvian circle
            naval_x  = int(bottom_x + dx * 2. * nc) 
            naval_y  = int(bottom_y + dy * 2. * nc)
            dx       = k[3,0,0] - naval_x 
            dy       = k[3,0,1] - naval_y 
            r        = int(sqrt(dx*dx+dy*dy))
            cv2.circle(img, (naval_x,naval_y), r, color, 1) 

            cv2.drawMarker(img, (bottom_x,bottom_y),  (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
            cv2.drawMarker(img, (naval_x, naval_y),   (0, 0, 255), cv2.MARKER_CROSS, 10, 1)

            # Vitruvian square
            dx = (self.k[1,0,0]-self.k[0,0,0])
            dy = (self.k[1,0,1]-self.k[0,0,1])
            target_angle = pi/2
            rotation = target_angle - atan2(-dy, dx)
            rotation = rotation - 2. * pi * floor((rotation + pi)/2./pi)
            l2= sqrt(dx*dx+dy*dy)
            cx= self.k[0,0,0]
            cy= self.k[0,0,1]
            trans_mat_rot = cv2.getRotationMatrix2D(center=(cx, cy), angle=degrees(-rotation), scale=1.0)
            srcPts = np.array( [
                [ [cx-l2, cy-l2] ],
                [ [cx+l2, cy-l2] ],
                [ [cx+l2, cy+l2] ],
                [ [cx-l2, cy+l2] ]
            ], dtype=np.float32)
            dstPts = cv2.transform(srcPts, m=trans_mat_rot)
            cv2.polylines(img, [np.int32(dstPts)], True, color, 1, cv2.LINE_AA)
                    
        elif self.type == objectTypes['person17']:
            # Ultra light pose, person skeleton
            joint_pairs = [ 
                ( 0,  1), ( 1,  3), (0,  2), (2,  4), ( 5,  6), ( 5,  7), ( 7,  9), 
                ( 6,  8), ( 8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (12, 14),
                (13, 15), (14, 16) ]                
            # draw segments
            for i in range(16):
                if self.hasVisibility():
                    if self.v[joint_pairs[i][0]] and self.v[joint_pairs[i][1]]:
                        cv2.line(img,
                        (k[joint_pairs[i][0],0,0], k[joint_pairs[i][0],0,1]), 
                        (k[joint_pairs[i][1],0,0], k[joint_pairs[i][1],0,1]),
                        (255, 0, 0), 2)

            # draw joints
            for i in range(17):
                if self.hasVisibility(): # we have visibility
                    if self.v[i]: # draw if visible
                        cv2.circle(img, (k[i,0,0], k[i,0,1]), 3, (0, 255, 0), -1)

        elif self.type == objectTypes['person39']:
            
            # https://developers.google.com/ml-kit/vision/pose-detection
            
            lines = (
                ( 0, 1), ( 0, 4), ( 1, 2), ( 2, 3),  (3, 7),
                ( 4, 5), ( 5, 6), ( 6, 8), ( 9,10), (11,12),
                (11,13), (11,23), (12,14), (12,24), (13,15),
                (14,16), (15,17), (15,19), (15,21), (16,18),
                (16,20), (16,22), (17,19), (18,20), (23,24))

            extended_lines_fb = (
                (23,25), (24,26), (25,27), (26,28), (27,29),
                (27,31), (28,30), (28,32), (29,31), (30,32))
            
            left_body = (
                ( 1, 2), ( 2, 3), ( 3, 4), ( 3, 7), (11,13),
                (13,15), (15,17), (17,19), (19,15), (15,21),
                (11,23), (23,25), (25,27), (27,29), (29,31), 
                (31,27))
            
            right_body = ( 
                ( 4, 5), ( 5, 6), ( 6, 8), (12,14), (14,16), 
                (16,18), (18,20), (20,16), (16,22), (12,24), 
                (24,26), (26,28), (28,30), (30,32), (32,28))
                        
            # skeleton 
              
            # lines
            ls=len(lines)
            for i in range(ls):
                if(self.v[lines[i][0]] and self.v[lines[i][1]]):
                    cv2.line(img, (k[lines[i][0],0,0], k[lines[i][0],0,1]),
                                    (k[lines[i][1],0,0], k[lines[i][1],0,1]),
                                self.color1, 2, 8, 0)                
            # extended lines
            ls=len(extended_lines_fb)
            for i in range(ls):
                if(self.v[extended_lines_fb[i][0]] and self.v[extended_lines_fb[i][1]]):
                    cv2.line(img, (k[extended_lines_fb[i][0],0,0], k[extended_lines_fb[i][0],0,1]),
                                    (k[extended_lines_fb[i][1],0,0], k[extended_lines_fb[i][1],0,1]),
                                self.color2, 2, 8, 0)         

            # left body
            ls=len(left_body)
            for i in range(ls):
                if(self.v[left_body[i][0]] and self.v[left_body[i][1]]):
                    cv2.line(img, (k[left_body[i][0],0,0], k[left_body[i][0],0,1]),
                                    (k[left_body[i][1],0,0], k[left_body[i][1],0,1]),
                                self.color3, 2, 8, 0)
                            
                    cv2.circle(img, (k[left_body[i][0],0,0], k[left_body[i][0],0,1]), 3,
                                self.color5, 1, cv2.LINE_AA, 0)
                    cv2.circle(img, (k[left_body[i][1],0,0], k[left_body[i][1],0,1]), 3,
                                self.color5, 1, cv2.LINE_AA, 0)

            # right body
            ls=len(right_body)
            for i in range(ls):
                if(self.v[right_body[i][0]] and self.v[right_body[i][1]]):
                    cv2.line(img, (k[right_body[i][0],0,0], k[right_body[i][0],0,1]),
                                    (k[right_body[i][1],0,0], k[right_body[i][1],0,1]),
                                self.color4, 2, 8, 0)         
                    cv2.circle(img, (k[right_body[i][0],0,0], k[right_body[i][0],0,1]), 3,
                                self.color5, 1, cv2.LINE_AA, 0)
                    cv2.circle(img, (k[right_body[i][1],0,0], k[right_body[i][1],0,1]), 3,
                                self.color5, 1, cv2.LINE_AA, 0)

        else:
            pass # can not draw object

        # bounding box, all object have bounding box
        if self.isRotated():
            cv2.polylines(img, [bb], True, color, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, 
                (bb[0,0,0], bb[0,0,1]),
                (bb[1,0,0], bb[1,0,1]),
                color )

    def drawRect(self, img, color=(0, 0, 0)):
        """ 
        Draw rectangle of object
        """
        # bounding box, all object have bounding box
        bb = np.int32(self.bb)
        if self.isRotated():
            cv2.polylines(img, [bb], True, color, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, 
                (bb[0,0,0], bb[0,0,1]),
                (bb[1,0,0], bb[1,0,1]),
                color )
            
    def printText(self, img, text, color=(0, 0, 0)):
        img_h, img_w = img.shape[:2]
        (Label_width, Label_height), BaseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        h = Label_height + BaseLine
        x = int(self.bb[0,0,0])
        y = int(self.bb[0,0,1])
        if y                   < 0:     y = 0
        elif y                 > img_h: y = img_h
        if x + Label_width     > img_w: x = img_w - Label_width
        elif x                 < 0:     x = 0
        cv2.rectangle(img,
            (x, y),
            (x + Label_width, y + h),
            (255,255,255), -1, )
        cv2.putText(img, text, (x, y + Label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, )

########################################################################################
# Draw Multiple Objects
########################################################################################
        
def drawObjects(img, objects, color=(255, 255, 255)):
    """ 
    Apply drawing function to all objects
    """
    if type(objects) is Object:
        objects.draw(img,color)
    elif type(objects) is list:
        for obj in objects:
            obj.draw(img, color)

########################################################################################
# Resize Objects
########################################################################################

def calculateBox(object, w:cython.int, h:cython.int, config):
    """    
    Adjusts object rectangle to model specifiations.
    It will create new rectangle centered on existing rectangle but shifted and scaled and with adjusted aspect ratio.
    The landmarks are not modified, only the rectangle is adjusted.
    Input:
      w,h: image size
      object: object location as computed for the image with size w x h
      config: cnn model configuration as shown below
    Output:
      new object with modified rectangle coordiantes matching aspect ratio of model specifications
      landmarks are returned with new object if available
    The new box will not exceed the image size.
    If the new box would extend past image dimensions, the box is shifted to remain within the image. 
    After completing this fucntion, one can extract the ROI and resize it to the model specs before feeding it into cnn.
    Example cnn model configuration:
    self.ModelConfig1 = {
        'scale'      : 2.7,       # <-- used to enlarge face box
        'shift_x'    : 0.0,       # <-- used to shift face box
        'shift_y'    : 0.0,       # <-- used to shift face box
        'height'     : 80,        # <-- used to change aspect ratio of face box. skipped if -1
        'width'      : 80,        # <-- used to change aspect ratio of face box. skipped if -1
    }
    This code allocates new face object.
    """
    # Origin: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing-APK/tree/master/engine/src/main/assets/live
            
    # Current Face Box Characteristics
    e = object.extent()
    box_width = e[0,0,0]
    box_height = e[0,0,1]
    box_aspect = box_width/box_height
    c = object.center()
    box_center_x = c[0,0,0]
    box_center_y = c[0,0,1]
    
    # New Facebox Characteristics
    # Aspect ratio:
    if (config['width'] == -1) or (config['height'] == -1): # keep original aspect ratio
        box_aspect_new = box_aspect 
        box_height_new = box_height
        box_width_new  = box_width
    else:
        box_aspect_new = config['width']/config['height'] # match aspect ratio to model
        
        # 4 possibilities for changeing aspect ratio
        if box_aspect < 1: # original box is taller than wide
            if box_aspect_new <= 1.0: # new box will also be taller than wide: keep original height
                box_height_new = box_height
                box_width_new  = box_height_new * box_aspect_new
            else: # new box will become wider than tall; keep original width
                box_width_new  = box_width
                box_height_new = box_width_new / box_aspect_new
        else: # original box is wider than tall
            if box_aspect_new <= 1.0: # new box will become taller than wide:  keep original width
                box_width_new  = box_width
                box_height_new = box_width_new / box_aspect_new
            else: # new box will become wider than tall: keep original height
                box_height_new = box_height
                box_width_new  = box_height_new * box_aspect_new

    # Scale Box
    # the desired box enlargement (scale) for the model limited by the image size
    scale = min( config['scale'], min( float(w)/(box_width_new), float(h)/box_height_new) ) 
    box_width_new  *= scale
    box_height_new *= scale 

    # Box shift
    if config['shift_x'] == 0:
        # New box corner coordinates    
        left_top_x     = (box_center_x - (box_width_new  / 2.))
        right_bottom_x = (box_center_x + (box_width_new  / 2.))
    else:
        shift_x = box_width  * config['shift_x'] # scale shift to pixels
        # New box corner coordinates    
        left_top_x     = (box_center_x - (box_width_new  / 2.) + shift_x)
        right_bottom_x = (box_center_x + (box_width_new  / 2.) + shift_x)
    
    if config['shift_y'] == 0:
        # New box corner coordinates    
        left_top_y     = (box_center_y - (box_height_new / 2.))
        right_bottom_y = (box_center_y + (box_height_new / 2.))
    else: 
        shift_y = box_height * config['shift_y'] # scale shift to pixels
        # New box corner coordinates
        left_top_y     = (box_center_y - (box_height_new / 2.) + shift_y)
        right_bottom_y = (box_center_y + (box_height_new / 2.) + shift_y)
    
    # Correct box to not exceed image with and height
    left_top_x     = clip(left_top_x, w-1)
    left_top_y     = clip(left_top_y, h-1)
    right_bottom_x = clip(right_bottom_x, w-1)
    right_bottom_y = clip(right_bottom_y, h-1)
        
    object_out = deepcopy(object)
    object_out.bb = np.array([
        [[left_top_x,     left_top_y]],
        [[right_bottom_x, right_bottom_y]],
        ], dtype=np.float32)
    
    return object_out

########################################################################################
# Smooth Landmarks
########################################################################################

class LandmarksSmoothingFilter: 
    '''
    From        
                https://github.com/geaxgx/depthai_blazepose/blob/main/mediapipe_utils.py
    Adapted from: 
                https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/landmarks_smoothing_calculator.cc
    
    frequency, min_cutoff, beta, derivate_cutoff: 
                See class OneEuroFilter description.
    min_allowed_object_scale:
                If calculated object scale is less than given value smoothing will be
                disabled and landmarks will be returned as is. Default=1e-6
    disable_value_scaling:
                Disable value scaling based on object size and use `1.0` instead.
                If not disabled, value scale is calculated as inverse value of object
                size. Object size is calculated as maximum side of rectangular bounding
                box of the object in XY plane. Default=False
    '''
    def __init__(self,
                frequency=30,
                min_cutoff=1,
                beta=0,
                derivate_cutoff=1,
                min_allowed_object_scale=1e-6,
                disable_value_scaling=False
                ):
        self.frequency                = frequency
        self.min_cutoff               = min_cutoff
        self.beta                     = beta
        self.derivate_cutoff          = derivate_cutoff
        self.min_allowed_object_scale = min_allowed_object_scale
        self.disable_value_scaling    = disable_value_scaling
        self.init                     = True

    @staticmethod
    def get_object_scale(landmarks):
        # Estimate object scale to use its inverse value as velocity scale for
        # RelativeVelocityFilter. If value will be too small (less than
        # `options_.min_allowed_object_scale`) smoothing will be disabled and
        # landmarks will be returned as is.
        # Object scale is calculated as average between bounding box width and height
        # with sides parallel to axis.
        min_xy = np.min(landmarks[:,:2], axis=0)
        max_xy = np.max(landmarks[:,:2], axis=0)
        return np.mean(max_xy - min_xy)

    def apply(self, landmarks, timestamp, object_scale=0):
        # object_scale: in practice, we use the size of the rotated rectangle region.rect_w_a=region.rect_h_a

        # Initialize filters 
        if self.init:
            self.filters = OneEuroFilter(self.frequency, self.min_cutoff, self.beta, self.derivate_cutoff)
            self.init = False

        # Get value scale as inverse value of the object scale.
        # If value is too small smoothing will be disabled and landmarks will be
        # returned as is.  
        if self.disable_value_scaling:
            value_scale = 1
        else:
            object_scale = object_scale if object_scale else self.get_object_scale(landmarks) 
            if object_scale < self.min_allowed_object_scale:
                return landmarks
            value_scale = 1 / object_scale

        return self.filters.apply(landmarks, value_scale, timestamp)

    def get_alpha(self, cutoff):
        '''
        te = 1.0 / self.frequency
        tau = 1.0 / (2 * Math.PI * cutoff)
        result = 1 / (1.0 + (tau / te))
        '''
        return 1.0 / (1.0 + (self.frequency / (2 * pi * cutoff)))

    def reset(self):
        self.init = True

class OneEuroFilter: 
    '''
    Adapted from: https://github.com/google/mediapipe/blob/master/mediapipe/util/filtering/one_euro_filter.cc
    Paper: https://cristal.univ-lille.fr/~casiez/1euro/
    frequency:  
                Frequency of incoming frames defined in seconds. Used
                only if can't be calculated from provided events (e.g.
                on the very first frame). Default=30
    min_cutoff:  
                Minimum cutoff frequency. Start by tuning this parameter while
                keeping `beta=0` to reduce jittering to the desired level. 1Hz
                (the default value) is a a good starting point.
    beta:       
                Cutoff slope. After `min_cutoff` is configured, start
                increasing `beta` value to reduce the lag introduced by the
                `min_cutoff`. Find the desired balance between jittering and lag. Default=0
    derivate_cutoff: 
                Cutoff frequency for derivate. It is set to 1Hz in the
                original algorithm, but can be turned to further smooth the
                speed (i.e. derivate) on the object. Default=1
    '''
    def __init__(self,
                frequency=30,
                min_cutoff=1,
                beta=0,
                derivate_cutoff=1,
                ):
        self.frequency       = frequency
        self.min_cutoff      = min_cutoff
        self.beta            = beta
        self.derivate_cutoff = derivate_cutoff
        self.x               = LowPassFilter(self.get_alpha(min_cutoff))
        self.dx              = LowPassFilter(self.get_alpha(derivate_cutoff))
        self.last_timestamp  = 0

    def get_alpha(self, cutoff):
        '''
        te = 1.0 / self.frequency
        tau = 1.0 / (2 * Math.PI * cutoff)
        result = 1 / (1.0 + (tau / te))
        '''
        return 1.0 / (1.0 + (self.frequency / (2 * pi * cutoff)))

    def apply(self, value, value_scale, timestamp):
        '''
        Applies filter to the value.
        timestamp in s associated with the value (for instance,
        timestamp of the frame where you got value from).
        '''
        if self.last_timestamp >= timestamp:
            # Results are unpreditable in this case, so nothing to do but return same value.
            return value

        # Update the sampling frequency based on timestamps.
        if self.last_timestamp != 0 and timestamp != 0:
            self.frequency = 1 / (timestamp - self.last_timestamp)
        self.last_timestamp = timestamp

        # Estimate the current variation per second.
        if self.x.has_last_raw_value():
            dvalue = (value - self.x.last_raw_value()) * value_scale * self.frequency
        else:
            dvalue = 0
        edvalue = self.dx.apply_with_alpha(dvalue, self.get_alpha(self.derivate_cutoff))

        # Use it to update the cutoff frequency
        cutoff = self.min_cutoff + self.beta * np.abs(edvalue)

        # filter the given value.
        return self.x.apply_with_alpha(value, self.get_alpha(cutoff))
        
class LowPassFilter:
    '''
    Adapted from: https://github.com/google/mediapipe/blob/master/mediapipe/util/filtering/low_pass_filter.cc
    Note that 'value' can be a numpy array
    '''
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.initialized = False

    def apply(self, value):
        if self.initialized:
            # Regular lowpass filter.
            # result = alpha * value + (1 - alpha) * stored_value;
            result = self.alpha * value + (1 - self.alpha) * self.stored_value
        else:
            result = value
            self.initialized = True
        self.raw_value = value
        self.stored_value = result
        return result

    def apply_with_alpha(self, value, alpha):
        self.alpha = alpha
        return self.apply(value)

    def has_last_raw_value(self):
        return self.initialized

    def last_raw_value(self):
        return self.raw_value

    def last_value(self):
        return self.stored_value

    def reset(self):
        self.initialized = False
