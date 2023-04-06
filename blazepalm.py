###########################################################
# Blazepaln Hand Detection
# Urs Utzinger, Spring 2023
###########################################################
# https://github.com/geaxgx/depthai_hand_tracker
# https://google.github.io/mediapipe/solutions/hands.html

import ncnn
import cython
import time
import numpy as np
import cv2
from utils_image import resizeImg
from utils_object import objectTypes
from utils_cnn import nms_cv, nms_weighted
from utils_blaze import decode_boxes, AnchorsParams, generate_anchors_np
from math import degrees

####################################################################
# Utility Functions
####################################################################
from scipy.special import expit # is same as sigmoid
def sigmoid_np(x):
    # if len(x) < 400: # is faster for small arrays
    # return(expit(x))
    # else: return(1. / (1. + np.exp(-x)))
    return (1. / (1. + np.exp(-x)))

###########################################################################
# Hand Detection
###########################################################################
# https://github.com/FeiGeChuanShu/ncnn-Android-mediapipe_hand
# https://github.com/vidursatija/BlazePalm
# https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md
# https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt

class Palm:
    
    def __init__(self, prob_threshold=0.4, nms_threshold=0.3, level='light', num_threads=-1, use_gpu=False, use_lightmode=True ):

        self.prob_threshold = prob_threshold
        self.nms_threshold  = nms_threshold
        self.num_threads    = num_threads
        self.use_gpu        = use_gpu
        self.lightmode      = use_lightmode
        
        self.ModelConfig = {
            'height'     : 192,
            'width'      : 192,
            'targetsize' : 192,
            'pad'        : True,
            'base'       : -1,
            'name'       : "blazehandpose_mediapipe/palm-full-op", # verified palm-lite-op or palm-full-op
            'mean'       : 0.,
            'std'        : 255.
        }

        if level == "light":
            self.ModelConfig['name'] = "blazehandpose_mediapipe/palm-lite-op"
        elif level == "full":
            self.ModelConfig['name'] = "blazehandpose_mediapipe/palm-full-op"
        
        self.m = (  self.ModelConfig['mean'],    self.ModelConfig['mean'],    self.ModelConfig['mean'])
        self.s = (1./self.ModelConfig['std'],  1./self.ModelConfig['std'],  1./self.ModelConfig['std'])

        self.handnet = ncnn.Net()
        self.handnet.clear()
        
        ncnn.set_cpu_powersave(2)
        ncnn.set_omp_num_threads(ncnn.get_big_cpu_count())        

        self.handnet.opt = ncnn.Option()
        self.handnet.opt.use_vulkan_compute = self.use_gpu
        self.handnet.opt.num_threads = ncnn.get_big_cpu_count()
                
        self.handnet.load_param("./models/" + self.ModelConfig['name'] +".param")
        self.handnet.load_model("./models/" + self.ModelConfig['name'] +".bin")

        # object that handles anchors
        self.anchor_options = AnchorsParams( 
            num_layers          = 4, 
            input_size_width    = self.ModelConfig['width'], 
            input_size_height   = self.ModelConfig['height'], 
            min_scale           = 0.1484375,  
            max_scale           = 0.75, 
            anchor_offset_x     = 0.5,  
            anchor_offset_y     = 0.5, 
            strides             = [8, 16, 16, 16], 
            aspect_ratios       = [1.0], 
            fixed_anchor_size   = True, 
            interpolated_scale_aspect_ratio=1.0, 
            reduce_boxes_in_lowest_layer=False)
        
        self.anchors = generate_anchors_np(self.anchor_options)
                        
    def __del__(self):
        self.net = None

    def __call__(self, img_bgr, scale=True, use_weighted_nms=False):
        '''
        Extract hands from image
        '''
        tic_start = time.perf_counter()

        (height, width) = img_bgr.shape[:2]
        # Scale image
        if scale:
            newWidth  = self.ModelConfig['width']
            newHeight = self.ModelConfig['height']
            img_rp, factor, l, t = resizeImg(img_bgr, newWidth, newHeight, pad=self.ModelConfig['pad'])  
            (height_rp, width_rp) = img_rp.shape[:2]
        else: 
            # Square image even if we dont scale
            newWidth  = max(width, height)
            newHeight = newWidth
            img_rp, factor, l, t = resizeImg(img_bgr, newWidth, newHeight, pad=self.ModelConfig['pad'])  
            (height_rp, width_rp) = img_rp.shape[:2]

        # convert to ncnn matrix
        mat_in = ncnn.Mat.from_pixels(img_rp, ncnn.Mat.PixelType.PIXEL_BGR2RGB, width_rp, height_rp)
        
        # Normalize
        mat_in.substract_mean_normalize(self.m, self.s)

        # Create an extractor
        ex = self.handnet.create_extractor() # every time a new extractor to clear internal caches
        if not (self.num_threads == -1): ex.set_num_threads(self.num_threads)
        ex.set_light_mode(self.lightmode)
        ex.input("input", mat_in)
        tic_prep = time.perf_counter()

        # Extract boxes and scores
        scores = ex.extract("cls")[1].numpy()
        bboxes = ex.extract("reg")[1].numpy()
        tic_extract = time.perf_counter()

        # Decode boxes
        # Has 7 keypoints
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
        #
        # in relative coordinates
        if use_weighted_nms:
            objects = nms_weighted(
                nms_threshold = self.nms_threshold,
                x0 = x0, y0 = y0,
                x1 = x1, y1 = y1,
                l  = np.ones(score.shape),
                p  = score,
                kx = keypoints_x, ky = keypoints_y,
                use_weighted=use_weighted_nms, # 0.79-0.85ms, 0.72-0.84ms cythonized            
                type=objectTypes['palm7'])
        else:        
            objects = nms_cv(
                nms_threshold = self.nms_threshold, 
                x0 = x0, y0 = y0, 
                x1 = x1, y1 = y1, 
                l  = np.ones(score.shape),
                p  = score, 
                kx = keypoints_x, ky = keypoints_y, 
                type=objectTypes['palm7']
            )
        tic_nms = time.perf_counter()

        # Create palm objects

        palms=[]
        for object in objects:
            # scale object to absolute size in relation of height_rp, width_rp
            object.relative2absolute(height=height_rp, width=width_rp)
            # Debug
            # object.draw(img_rp, color=(0,0,0))
            # cv2.imshow('Debug', img_rp)    
            # cv2.waitKey()
            #
            # scale object to original image
            object.resize(scale=factor, l=l, t=t)  

            # keypoints 
            # 0 = center of wrist
            # 1 = start of pointing finger
            # 2 = start of middle finger
            # 3 = start of ring finger
            # 4 = start of pinky finger
            # 5 = start of thumb
            # 6 = 2nd thumb joint
            #
            # length of plam is keypoint 0 to keypoint 2
            # length of finger is length of palm
            # center of palm is half between keypoint 0 and 2
            # length of hand object is center of palm +/- 1.5*length of palm
            # center of hand is start of middle finger
            # width of hand is width of bounding box
            # rotation angle is from center of wrist to center of middle finger versus vertical
            # rotation around center of hand
            
            # center_palm = (object.k[0,:,:] + object.k[2,:,:]) / 2
            center_hand = object.k[2,:,:]
            l2  = cv2.norm(object.k[0,:,:] - object.k[1,:,:])
            ptl = center_hand - l2
            ptr = center_hand + np.array([l2, -l2])
            pbr = center_hand + l2
            pbl = center_hand + np.array([-l2, l2])
            bb  = np.array( [ptl, ptr, pbr, pbl], dtype=np.float32)
            # Rotate the bounding box
            #   caculate rotations -pi .. pi
            rotation = object.angle()
            trans_mat_rot = cv2.getRotationMatrix2D(center=center_hand[0], angle=degrees(-rotation), scale=1.4)
            object.bb = cv2.transform(bb, m=trans_mat_rot)
            palms.append(object)
            
        tic_createpalm = time.perf_counter()

        return palms, \
               np.array([tic_prep-tic_start, 
                         tic_extract-tic_prep,
                         tic_decode-tic_extract, 
                         tic_nms-tic_decode, 
                         tic_createpalm-tic_nms])

###########################################################################
# Main Testing
###########################################################################
# Origin: Urs Utzinger

if __name__ == '__main__':
    import cv2
    import time
    from blazepalm import Palm
    
    # img = cv2.imread('images/oath.jpg')
    img = cv2.imread('images/soccerplayer.jpg')
    # img = cv2.imread('images/zidane.jpg')
    # img = cv2.imread('images/Pic_Team1.jpg')
    # (m,n) = img.shape[:2]
    # img = img[int(m*0.25):int(m*0.75),int(n*.25):int(n*0.75),:]
    # img = cv2.imread('images/hand.jpg')
    # (height, width) = img.shape[:2]
    # hand_r = cv2.copyMakeBorder(src=img, top=50, bottom=50, left=50, right=50, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))
    # hand_l = cv2.flip(hand_r.copy(), 1)
    # img = np.hstack((hand_l,hand_r))

    if img is None: print('Error opening image')
    
    # find hands, low threshold shows performance of algorithm, make it large for real world application
    # level is full or light
    palmnet = Palm(prob_threshold=0.55, nms_threshold=0.3, level='full', use_gpu=False)
            
    palms, times_hand = palmnet(img, scale=True, use_weighted_nms=False)

    for palm in palms:
        palm.draw(img, color=(0,0,0))

    cv2.imshow('Hands', img)    
    cv2.waitKey()

    print("Preprocess  {:.2f} ms".format(1000.*times_hand[0]))
    print("Extract     {:.2f} ms".format(1000.*times_hand[1]))
    print("Decode      {:.2f} ms".format(1000.*times_hand[2]))
    print("Select NMS  {:.3f} ms".format(1000.*times_hand[3]))
    print("Create Palm {:.3f} ms".format(1000.*times_hand[4]))

    handnet = None
    cv2.destroyAllWindows()