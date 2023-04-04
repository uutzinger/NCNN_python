###########################################################################
# Person Pose Class
###########################################################################
# Google Pose Detection
# https://google.github.io/mediapipe/solutions/pose
# https://paperswithcode.com/paper/blazepose-on-device-real-time-body-pose
# 
#   https://developers.google.com/ml-kit/vision/pose-detection
#   https://ai.googleblog.com/2020/08/on-device-real-time-body-pose-tracking.html
#   https://github.com/google/mediapipe/tree/master/mediapipe/modules/pose_landmark
# Code:
# https://github.com/FeiGeChuanShu/ncnn_Android_BlazePose (2022) #####
# https://github.com/axinc-ai/ailia-models/blob/master/pose_estimation_3d/blazepose-fullbody/blazepose_utils.py
##########################################################################
# Alternative Pose Detection
# https://github.com/dog-qiuqiu/Ultralight-SimplePose (2022)
# https://github.com/FeiGeChuanShu/ncnn_Android_MoveNet  (2022)
##########################################################################

# Pose landmarks within the given ROI. (NormalizedLandmarkList)
# We have 33 landmarks (see pose_landmark_topology.svg) and there are other auxiliary key points.
#  0 - nose
#  1 - left eye (inner)
#  2 - left eye
#  3 - left eye (outer)
#  4 - right eye (inner)
#  5 - right eye
#  6 - right eye (outer)
#  7 - left ear
#  8 - right ear
#  9 - mouth (left)
# 10 - mouth (right)
# 11 - left shoulder
# 12 - right shoulder
# 13 - left elbow
# 14 - right elbow
# 15 - left wrist
# 16 - right wrist
# 17 - left pinky
# 18 - right pinky
# 19 - left index
# 20 - right index
# 21 - left thumb
# 22 - right thumb
# 23 - left hip
# 24 - right hip
# 25 - left knee
# 26 - right knee
# 27 - left ankle
# 28 - right ankle
# 29 - left heel
# 30 - right heel
# 31 - left foot index
# 32 - right foot index

import ncnn
import cython
import time
import numpy as np
import cv2
from utils_image import resizeImg, extractObjectROI
from utils_object import objectTypes, Object

from scipy.special import expit # is same as sigmoid
def sigmoid_np(x):
    # if len(x) < 400:
    # use expit(x))
    # else: (1. / (1. + np.exp(-x)))
    return (1. / (1. + np.exp(-x)))
  
###########################################################################
# Skelegon Estimation
###########################################################################

class PersonLandmarkDetect:
    
    def __init__(self, prob_threshold=0.4, nms_threshold=0.5, vis_threshold=0.5, level="light", num_threads=-1, use_gpu=False, use_lightmode=True):

        self.prob_threshold = prob_threshold
        self.nms_threshold  = nms_threshold
        self.vis_threshold  = vis_threshold
        self.num_threads    = num_threads
        self.use_gpu        = use_gpu
        self.lightmode      = use_lightmode
        
        # The ncnn model is from
        # https://github.com/FeiGeChuanShu/ncnn_Android_BlazePose/tree/main/app/src/main/jni
        
        self.ModelConfig = {
            'height'     : 256,
            'width'      : 256,
            'targetsize' : 256,
            'pad'        : True,
            'base'       : -1,
            'name'       : "blazepersonpose/lite",            # lite, full or heavy
            'mean'       : 0.,                          #
            'std'        : 255.,                        # 
        }

        if level == "light":
            self.ModelConfig['name'] = "blazepersonpose/lite"
        elif level == "full":
            self.ModelConfig['name'] = "blazepersonpose/full"
        elif level == "heavy": 
            self.ModelConfig['name'] = "blazepersonpose/heavy"

        self.m = (  self.ModelConfig['mean'],  self.ModelConfig['mean'],   self.ModelConfig['mean'])
        self.s = (1./self.ModelConfig['std'],1./self.ModelConfig['std'],1./self.ModelConfig['std'])

        self.personlandmark = ncnn.Net()
        self.personlandmark.clear()

        ncnn.set_cpu_powersave(2)
        ncnn.set_omp_num_threads(ncnn.get_big_cpu_count())        
        
        self.personlandmark.opt = ncnn.Option()
        self.personlandmark.opt.use_vulkan_compute = self.use_gpu
        self.personlandmark.opt.num_threads = ncnn.get_big_cpu_count()
        
        self.personlandmark.load_param("./models/" + self.ModelConfig['name'] +".param")
        self.personlandmark.load_model("./models/" + self.ModelConfig['name'] +".bin")
              
    def __del__(self):
        self.net = None

    def __call__(self, img_bgr):
        '''
        Extract pose landmarks from a pose image
        '''
        tic_start = time.perf_counter()

        # Prepare NCNN input
        #
        (height, width) = img_bgr.shape[:2]                
        # No scaling if the image matches model size
        if height == self.ModelConfig['height'] and width == self.ModelConfig['width']:
            img_rp = img_bgr
            scaled = False
        else:
            img_rp, factor, l, t = resizeImg(img_bgr, self.ModelConfig['widtht'], self.ModelConfig['height'], pad=self.ModelConfig['pad'])  
            scaled = True
        (height_rp, width_rp) = img_rp.shape[:2]    
        # Convert to ncnn matrix
        mat_in = ncnn.Mat.from_pixels(img_rp, ncnn.Mat.PixelType.PIXEL_BGR2RGB, width_rp, height_rp)
        # Normalize
        mat_in.substract_mean_normalize(self.m, self.s)
 
        # Create an extractor
        ex = self.personlandmark.create_extractor() # every time a new extractor to clear internal caches
        if not (self.num_threads == -1): ex.set_num_threads(self.num_threads)
        ex.set_light_mode(self.lightmode)
        ex.input("input", mat_in)
        tic_prep = time.perf_counter()
        
        # Extract overall pose score
        poseflag = ex.extract("poseflag")[1].numpy()[0]
        
        # Found Pose, extract keypoints
        if(poseflag >= 0.5): 
            ld_3d = ex.extract("ld_3d")[1].numpy()
            tic_extract = time.perf_counter()

            # Keypoints
            # num_kpt = 39
            # num_kpt_offset = int(len(ld_3d) / num_kpt)
            kpt_offset = 5
            keypoints_x = ld_3d[0::kpt_offset]
            keypoints_y = ld_3d[1::kpt_offset]
            keypoints_z = ld_3d[2::kpt_offset]
            visibility  = ld_3d[3::kpt_offset]
            presence    = ld_3d[4::kpt_offset]
            # Make occluded points invisible
            keypoints_v = sigmoid_np(np.minimum(visibility,presence)) >= self.vis_threshold

            # 39 keypoints
            # 0..32, not sure what the last 6 points represent, they are on the hands and on top of head
            k = np.array( [
                [keypoints_x],
                [keypoints_y],
                [keypoints_z]
            ], dtype=np.float32).T            

            object = Object(
                type = objectTypes['person39'], # object type 
                bb = np.array( [    # bounding box
                    [ [0,          0          ] ], # top left
                    [ [width_rp-1, 0          ] ], # top right
                    [ [width_rp-1, height_rp-1] ], # bottom right
                    [ [0,          height_rp-1] ]  # bottom left
                ], dtype=np.float32),  
                p  = poseflag,      # probability
                l  = -1,            # label number
                k  = k,             # keypoints
                v  = keypoints_v)   # visibility of keypoints
            
            # scale object to original image
            if scaled: object.resize(scale=factor, l=l, t=t)
            
            tic_skeleton = time.perf_counter()
        else:
            object = None
            tic_extract = time.perf_counter()
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
    from blazeperson import Person
    from blazepose import PersonLandmarkDetect
    
    # img = cv2.imread('images/person.jpg')
    # img = cv2.imread('images/girl-5204299_640.jpg')
    # img = cv2.imread('images/pexels-yogendra-singh-1701205.jpg')
    # img = cv2.imread('images/pexels-yogendra-singh-1701202.jpg')
    # img = cv2.imread('images/pexels-yogendra-singh-1701193.jpg')
    # img = cv2.imread('images/pexels-yogendra-singh-1701206.jpg')
    img = cv2.imread('images/pexels-yogendra-singh-3307487.jpg')
    (height, width) = img.shape[:2]
    scale = 0.125
    h = int(height* scale)
    w = int(width * scale)
    img = cv2.resize(img, (w, h), cv2.INTER_LINEAR)

    if img is None: print('Error opening image')
    
    # find faces, low threshold shows performance of algorithm, make it large for real world application
    personnet   = Person(prob_threshold=0.5, nms_threshold=0.3, use_gpu=False)
    # light, full, heavy
    skeletonnet = PersonLandmarkDetect(prob_threshold=0.5, nms_threshold=0.3, vis_threshold=0.1, level="light", use_gpu=False)
    
    # Find person
    persons, portraits, times_person = personnet(img, scale=True)

    # times_landmark = [0,0,0]
    for (person, portrait) in zip(persons, portraits):

        img_aligned, mat_trans, rotatedObject = extractObjectROI(img, person, target_size = 256, simple=True)
        person_pose, times_landmark =  skeletonnet(img_aligned)

        # Debug display on original image
        # person.draw(img)
        # portrait.drawRect(img, color=(0,255,0))
        # cv2.imshow('Debug', img)    
        # cv2.waitKey(0)
        # Debug display on ROI
        # person_pose.draw(img_aligned)
        # cv2.imshow('Debug', img_aligned)    
        # cv2.waitKey(0)

        # Display on original image
        person_pose.invtransform(mat_trans)
        person_pose.draw(img)
        portrait.drawRect(img, color=(0,255,0))
        
    cv2.imshow('Person Pose', img)    
    cv2.waitKey()
   
    print("Preprocess    {:.2f} ms".format(1000.*times_person[0]))
    print("Extract       {:.2f} ms".format(1000.*times_person[1]))
    print("Decode        {:.2f} ms".format(1000.*times_person[2]))
    print("Select NMS    {:.3f} ms".format(1000.*times_person[3]))
    print("Create Person {:.3f} ms".format(1000.*times_person[4]))

    print("Preprocess    {:.2f} ms".format(1000.*times_landmark[0]))
    print("Extract       {:.2f} ms".format(1000.*times_landmark[1]))
    print("Skeleton      {:.3f} ms".format(1000.*times_landmark[2]))

    personnet = None
    landmarknet = None