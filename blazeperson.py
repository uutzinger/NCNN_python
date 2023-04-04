###########################################################################
# Blaze Person Detector
# https://github.com/FeiGeChuanShu/ncnn_Android_BlazePose (2022)
###########################################################################
# https://google.github.io/mediapipe/solutions/pose
# https://developers.google.com/ml-kit/vision/pose-detection/android
# https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html
# https://google.github.io/mediapipe/solutions/pose
# https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
# https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_landmarks_to_roi.pbtxt
# https://paperswithcode.com/paper/blazepose-on-device-real-time-body-pose
# https://github.com/axinc-ai/ailia-models/blob/master/pose_estimation_3d/blazepose-fullbody/blazepose_utils.py
###########################################################################
# Keypoints based on Vitruvian man
# https://en.wikipedia.org/wiki/Vitruvian_Man#/media/File:Da_Vinci_Vitruve_Luc_Viatour.jpg
# Extended arms =  height of man
# Navel from bottom of feet (sqrt(5/4) -1/2) * height of man
#   taken from figure 9 of https://link.springer.com/article/10.1007/s00004-015-0247-7
#   sqrt(h/2*h/2 + h*h) - h/2 which is (sqrt(5/4) - 1/2) * h
#
# Key point 0 - mid hip center
# Key point 1 - point that encodes size & rotation (for full body) (top of head)
# Key point 2 - mid shoulder center
# Key point 3 - point that encodes size & rotation (for upper body) (vitruvian circle goes through this point)
# 
# Height of man is 2 * Keypoint 0 to Keypoint 1
# Bottom of man is extension of keypoint 0 to keypoint 1 by half of height
#
# Operates on 224x224 image
# max/min -1/1
# uses keypoint 0 and 1 to determine rotation.
# Bounding Box is scaled by 1.25 and square long is used as size
#########################################################################################################
# Other code options for person detection and pose estimations are
# https://github.com/dog-qiuqiu/Ultralight-SimplePose (2022)
# https://github.com/FeiGeChuanShu/ncnn_Android_MoveNet  (2022)
# https://github.com/FeiGeChuanShu/ncnn_Android_BlazePose (2022)
#########################################################################################################

import ncnn
import cython
import time
import numpy as np
import cv2
from utils_image import resizeImg
from utils_object import objectTypes
from utils_blaze import decode_boxes, AnchorsParams, generate_anchors_np
from utils_cnn import nms_cv
from math import degrees
from copy import deepcopy

from scipy.special import expit # is same as sigmoid
def sigmoid_np(x):
    # if len(x) < 400: # is faster for small arrays
    # return(expit(x))
    # else: return(1. / (1. + np.exp(-x)))
    return (1. / (1. + np.exp(-x)))

class Person:

    def __init__(self, prob_threshold=0.5, nms_threshold=0.3, num_threads=-1, use_gpu=False, use_lightmode=True ):

        self.prob_threshold = prob_threshold
        self.nms_threshold  = nms_threshold
        self.num_threads    = num_threads
        self.use_gpu        = use_gpu
        self.lightmode      = use_lightmode

        # Original model is from 
        #    https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
        # The ncnn model likely is adapted from from https://github.com/nihui/ncnn-android-nanodet
            
        self.ModelConfig = {
            'height'     : 224,
            'width'      : 224,
            'base'       : -1,
            'pad'        : True,
            'name'       : "blazepose/detection",
            'mean'       : 127.5,
            'std'        : 127.5,
        }
        
        self.m = (self.ModelConfig['mean'], self.ModelConfig['mean'], self.ModelConfig['mean'])
        self.s = (1./self.ModelConfig['std'],  1./self.ModelConfig['std'],  1./self.ModelConfig['std'])
    
        ncnn.set_cpu_powersave(2)
        ncnn.set_omp_num_threads(ncnn.get_big_cpu_count())        

        self.personnet = ncnn.Net()
        self.personnet.opt.use_vulkan_compute = self.use_gpu
        self.personnet.opt.num_threads = ncnn.get_big_cpu_count()
        
        # The ncnn model is from
        self.personnet.load_param("./models/" + self.ModelConfig['name'] +".param")
        self.personnet.load_model("./models/" + self.ModelConfig['name'] +".bin")
        
        # ssd anchors
        self.anchor_options = AnchorsParams(
            num_layers                      = 5,
            min_scale                       = 0.1484375,
            max_scale                       = 0.75,
            input_size_height               = self.ModelConfig['height'],
            input_size_width                = self.ModelConfig['width'],
            anchor_offset_x                 = 0.5,
            anchor_offset_y                 = 0.5,
            strides                         = [8,16,32,32,32],
            aspect_ratios                   = [1.0],
            fixed_anchor_size               = True,
            interpolated_scale_aspect_ratio =  1.0,
            reduce_boxes_in_lowest_layer    = False
        )
        self.anchors = generate_anchors_np(self.anchor_options)
 
    def __del__(self):
        self.net = None

    def __call__(self, img_bgr, scale=True):
        '''
        Person detection per frame
        '''
            
        height:     cython.int
        width:      cython.int
        l:          cython.int
        t:          cython.int
        factor:     cython.double
        
        tic_start = time.perf_counter()

        (height, width) = img_bgr.shape[:2]
        
        # Scale image
        if scale:
            newWidth  = self.ModelConfig['width']
            newHeight = self.ModelConfig['height']
        else: 
            # Square image even if we dont scale
            newWidth  = max(width, height)
            newHeight = newWidth

        # convert to ncnn mat
        img_rp, factor, l, t = resizeImg(img_bgr, newWidth, newHeight, pad=self.ModelConfig['pad'])  
        (height_rp, width_rp) = img_rp.shape[:2]
        assert height_rp == width_rp # we need square image
        mat_in = ncnn.Mat.from_pixels(img_rp, ncnn.Mat.PixelType.PIXEL_BGR2RGB, width_rp, height_rp)
        
        # Normalize
        mat_in.substract_mean_normalize(self.m, self.s)
 
        # Create an extractor
        ex = self.personnet.create_extractor() # every time a new extractor to clear internal caches
        if not (self.num_threads == -1): ex.set_num_threads(self.num_threads)
        ex.set_light_mode(self.lightmode)
        ex.input("input", mat_in)
        tic_prep = time.perf_counter()

        # Extract boxes and scores
        scores = ex.extract("cls")[1].numpy()
        bboxes = ex.extract("reg")[1].numpy()
        tic_extract = time.perf_counter()
       
        # Decode boxes
        # Has 4 keypoints along line which is parallel to spine: 
        # [middle hip, top of head, middle sholder, above head]

        (x0, y0, x1, y1, keypoints_x, keypoints_y, score) = decode_boxes(
            score_thresh=self.prob_threshold, 
            input_img_w=width_rp, 
            input_img_h=height_rp, 
            raw_scores=scores, 
            raw_boxes=bboxes, 
            anchors = self.anchors, 
            anchors_fixed=self.anchor_options.fixed_anchor_size
        )
        tic_decode = time.perf_counter()

        # Non Maximum Suppression
        # in relative coordinates
        objects = nms_cv(
            nms_threshold = self.nms_threshold, 
            x0=x0, 
            y0=y0, 
            x1=x1, 
            y1=y1, 
            l=np.ones(score.shape),
            p=score, 
            kx=keypoints_x, 
            ky=keypoints_y, 
            v=[], 
            type=objectTypes['person4']
        )
        tic_nms = time.perf_counter()

        rotatedObjects=[]
        rotatedPortraits=[]
        for object in objects:
            # scale object to absolute size in relation of height_rp, width_rp
            object.relative2absolute(height=height_rp, width=width_rp)
            # scale object to original image
            object.resize(scale=factor, l=l, t=t)

            # Object with original bounding box
            portrait = deepcopy(object)
            rotation= object.angle()

            # Object with full body bounding box
            # the bounding box created by CNN is portrait only
            # Use the keypoints to calculate a bounding box, donot use the bounding box given which includes only head and torso
            # mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
            # compute box center and scale from landmark 0 and 1 (middle of hip and top  of head)
            center_person = object.k[0,:,:]
            l2  = cv2.norm(object.k[0,:,:] - object.k[1,:,:])
            ptl = center_person - l2
            ptr = center_person + np.array([l2, -l2])
            pbr = center_person + l2
            pbl = center_person + np.array([-l2, l2])
            # Replace bounding box
            bb = np.array( [ptl, ptr, pbr, pbl], dtype=np.float32)
            #   Rotate the bounding box
            trans_mat_rot = cv2.getRotationMatrix2D(center=center_person[0], angle=degrees(-rotation), scale=1.2)
            object.bb = cv2.transform(bb, m=trans_mat_rot)

            center_portrait = portrait.center()[0]
            l2 = np.max(portrait.bb[1,:,:] - portrait.bb[0,:,:])/2.0
            ptl = center_portrait - l2
            ptr = center_portrait + np.array([l2, -l2])
            pbr = center_portrait + l2
            pbl = center_portrait + np.array([-l2, l2])
            bb = np.array( [ptl, ptr, pbr, pbl], dtype=np.float32)
            trans_mat_rot = cv2.getRotationMatrix2D(center=center_portrait[0], angle=degrees(-rotation), scale=1.0)
            portrait.bb = cv2.transform(bb, m=trans_mat_rot)
                                        
            rotatedPortraits.append(portrait)
            rotatedObjects.append(object)
            
        tic_createperson = time.perf_counter()

        return rotatedObjects, rotatedPortraits,\
               np.array([tic_prep-tic_start, 
                         tic_extract-tic_prep,
                         tic_decode-tic_extract, 
                         tic_nms-tic_decode, 
                         tic_createperson-tic_nms])
    
###########################################################################
# Main Testing
###########################################################################
# Origin: Urs Utzinger

if __name__ == '__main__':
    import cv2
    import numpy as np
    import time
    from blazeperson import Person
    
    # img = cv2.imread('images/oath.jpg')
    # img = cv2.imread('images/soccerplayer.jpg')
    # img = cv2.imread('images/zidane.jpg')
    # img = cv2.imread('images/Pic_Team1.jpg')
    # m,n = img.shape[:2]
    # img = img[0:int(m/2), int(n/4):int(n*3/4), :]
    img = cv2.imread('images/person.jpg')
    # img = cv2.imread('images/girl-5204299_640.jpg')
    # img = cv2.imread('images/pexels-yogendra-singh-1701205.jpg')
    # img = cv2.imread('images/pexels-yogendra-singh-1701202.jpg')
    # img = cv2.imread('images/pexels-yogendra-singh-1701193.jpg')
    # img = cv2.imread('images/pexels-yogendra-singh-1701206.jpg')
    # img = cv2.imread('images/pexels-yogendra-singh-3307487.jpg')
    # (height, width) = img.shape[:2]
    # scale = 0.125
    # h = int(height* scale)
    # w = int(width * scale)
    # img = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
    
    #
    if img is None: print('Error opening image')

    img_display = img.copy()
        
    # find faces, low threshold shows performance of algorithm, make it large for real world application
    personnet = Person(prob_threshold=0.3, nms_threshold=0.8, use_gpu=False)
    
    persons, portraits, times_person = personnet(img, scale=True)

    for person, portrait in zip(persons, portraits):
        person.draw(img_display)
        portrait.drawRect(img_display, color=(0,255,0))
        # img_roi, trans_mat, rotatedObject = extractObjectROI(img_bgr, rotatedObject, target_size, sinmple=true)

    cv2.imshow('Blaze Person Object', img_display)    
    cv2.waitKey()

    print("Preprocess    {:.2f} ms".format(1000.*times_person[0]))
    print("Extract       {:.2f} ms".format(1000.*times_person[1]))
    print("Decode        {:.2f} ms".format(1000.*times_person[2]))
    print("Select NMS    {:.3f} ms".format(1000.*times_person[3]))
    print("Create Person {:.3f} ms".format(1000.*times_person[4]))

    personnet = None
    cv2.destroyAllWindows()