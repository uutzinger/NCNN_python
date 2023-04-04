####################################################################
# Blaze Hand Skeleton
####################################################################
# https://github.com/geaxgx/depthai_hand_tracker

import ncnn
import cython
import time
import numpy as np
import cv2
from utils_image import resizeImg, extractObjectROI
from utils_object import Object, objectTypes

from scipy.special import expit # is same as sigmoid
def sigmoid_np(x):
    # if len(x) < 400:
    # use expit(x))
    # else: (1. / (1. + np.exp(-x)))
    return (1. / (1. + np.exp(-x)))

###########################################################################
# Skeleton Estimation
###########################################################################

class HandLandmarkDetect:
    
    def __init__(self, prob_threshold=0.4, level='light', num_threads=-1, use_gpu=False, use_lightmode=True ):

        self.prob_threshold = prob_threshold
        self.num_threads    = num_threads
        self.use_gpu        = use_gpu
        self.lightmode      = use_lightmode
        
        # The ncnn model is from
        # https://github.com/FeiGeChuanShu/ncnn-Android-mediapipe_hand
        
        self.ModelConfig = {
            'height'     : 224,
            'width'      : 224,
            'targetsize' : 224,
            'pad'        : True,
            'base'       : -1,
            'name'       : "blazehandpose/hand_full-op", # verified hand_lite-op or hand_full-op
            'mean'       : 0.,
            'std'        : 255.
        }

        if level == "light":
            self.ModelConfig['name'] = "blazehandpose_mediapipe/hand_lite-op"
        elif level == "full":
            self.ModelConfig['name'] = "blazehandpose_mediapipe/hand_full-op"

        self.m = (  self.ModelConfig['mean'],  self.ModelConfig['mean'],   self.ModelConfig['mean'])
        self.s = (1./self.ModelConfig['std'],1./self.ModelConfig['std'],1./self.ModelConfig['std'])

        self.handlandmark = ncnn.Net()
        self.handlandmark.clear()

        ncnn.set_cpu_powersave(2)
        ncnn.set_omp_num_threads(ncnn.get_big_cpu_count())        
        
        self.handlandmark.opt = ncnn.Option()
        self.handlandmark.opt.use_vulkan_compute = self.use_gpu
        self.handlandmark.opt.num_threads = ncnn.get_big_cpu_count()
        
        self.handlandmark.load_param("./models/" + self.ModelConfig['name'] +".param")
        self.handlandmark.load_model("./models/" + self.ModelConfig['name'] +".bin")
              
    def __del__(self):
        self.net = None

    def __call__(self, img_bgr):
        '''
        Extract hand landmarks from a hand image
        '''
        tic_start = time.perf_counter()

        # Extract image 
        (height, width) = img_bgr.shape[:2]
        if height == self.ModelConfig['height'] and width == self.ModelConfig['width']:
            img_rp = img_bgr
            scaled = False
        else:
            img_rp, factor, l, t = resizeImg(img_bgr, self.ModelConfig['widtht'], self.ModelConfig['height'], pad=self.ModelConfig['pad'])  
            scaled = True
        (height_rp, width_rp) = img_rp.shape[:2]    

        # Convert to ncnn matrix
        mat_in = ncnn.Mat.from_pixels(img_bgr, ncnn.Mat.PixelType.PIXEL_BGR2RGB, width_rp, height_rp)
        
        # Normalize
        mat_in.substract_mean_normalize(self.m, self.s)
 
        # Create an extractor
        ex = self.handlandmark.create_extractor() # every time a new extractor to clear internal caches
        if not (self.num_threads == -1): 
            ex.set_num_threads(self.num_threads)
        ex.set_light_mode(self.lightmode)
        ex.input("input", mat_in)
        tic_prep = time.perf_counter()
        
        points = ex.extract("points")[1].numpy()
        score = ex.extract("score")[1].numpy()
        tic_extract = time.perf_counter()
        
        # i = np.arange(0, 21, 1)
        # keypoints_x = points[0,i * 3]
        # keypoints_y = points[0,i * 3 + 1]
        # keypoints_z = points[0,i * 3 + 2]

        kpt_offset = 3
        keypoints_x = points[0,0::kpt_offset]
        keypoints_y = points[0,1::kpt_offset]
        keypoints_z = points[0,2::kpt_offset]
        
        # 21 keypoints
        k = np.array( [
            [keypoints_x],
            [keypoints_y],
            [keypoints_z]
        ], dtype=np.float32).T            
        
        object = Object(
            type = objectTypes['hand21'], # object type 
            bb = np.array( [    # bounding box
                [ [0,          0          ] ], # top left
                [ [width_rp-1, 0          ] ], # top right
                [ [width_rp-1, height_rp-1] ], # bottom right
                [ [0,          height_rp-1] ]  # bottom left
            ], dtype=np.float32),  
            p  = score[0,0],    # probability
            l  = -1,            # label number
            k  = k,             # keypoints
            v  = [])            # visibility of keypoints
            # scale object to original image
        
        if scaled: object.resize(scale=factor, l=l, t=t)

        tic_skeleton = time.perf_counter()

        return object, [(tic_prep-tic_start), (tic_extract-tic_prep), (tic_skeleton - tic_extract)]
        
###########################################################################
# Main Testing
###########################################################################
# Origin: Urs Utzinger

if __name__ == '__main__':
    import cv2
    import numpy as np
    import time
    from blazepalm import Palm
    from blazehandpose import HandLandmarkDetect
    
    # img = cv2.imread('images/oath.jpg')
    # img = cv2.imread('images/soccerplayer.jpg')
    # img = cv2.imread('images/zidane.jpg')
    # img = cv2.imread('images/Pic_Team1.jpg')
    # (m,n) = img.shape[:2]
    # img = img[int(m*0.25):int(m*0.75),int(n*.25):int(n*0.75),:]
    img = cv2.imread('images/hand.jpg')
    (height, width) = img.shape[:2]
    hand_r = cv2.copyMakeBorder(src=img, top=50, bottom=50, left=50, right=50, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))
    hand_l = cv2.flip(hand_r.copy(), 1)
    img = np.hstack((hand_l,hand_r))

    if img is None: print('Error opening image')
    
    # find hands, low threshold shows performance of algorithm, make it large for real world application
    palmnet = Palm(prob_threshold=0.55, nms_threshold=0.3, level='full', use_gpu=False)
    landmarknet = HandLandmarkDetect(prob_threshold=0.55, level='full', use_gpu=False)
    
    hands, times_hand = palmnet(img, scale=True, use_weighted_nms=False)


    for hand in hands:
        img_aligned, mat_trans, rotatedObject = extractObjectROI(img, hand, target_size = 224, simple=True)
        hand_pose, times_landmark =  landmarknet(img_aligned)

        # Display on original image
        hand_pose.invtransform(mat_trans)
        hand_pose.draw(img)

        hand.draw(img)

    cv2.imshow('Hands', img)    
    cv2.waitKey()

    print("Preprocess  {:.2f} ms".format(1000.*times_hand[0]))
    print("Extract     {:.2f} ms".format(1000.*times_hand[1]))
    print("Decode      {:.2f} ms".format(1000.*times_hand[2]))
    print("Select NMS  {:.3f} ms".format(1000.*times_hand[3]))
    print("Create Palm {:.3f} ms".format(1000.*times_hand[4]))

    print("Preprocess  {:.2f} ms".format(1000.*times_landmark[0]))
    print("Extract     {:.2f} ms".format(1000.*times_landmark[1]))
    print("Skeleton    {:.3f} ms".format(1000.*times_landmark[2]))

    handnet = None
    landmarknet = None
    cv2.destroyAllWindows()