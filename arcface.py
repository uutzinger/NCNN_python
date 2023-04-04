import numpy as np
import ncnn
import cv2
from utils_image import extractObjectROI
import time


# Implementations for face recognition
# https://github.com/Qengineering/Face-Recognition-Raspberry-Pi-64-bits
# https://github.com/rainfly123/mobilefacenet-ncnn/
# https://github.com/XinghaoChen9/LiveFaceReco_RaspberryPi

# ARCFace
# https://arxiv.org/abs/1801.07698
# https://www.arxiv-vanity.com/papers/1801.07698/#:~:text=As%20most%20of%20the%20convolutional,224%C3%97224%20or%20larger.
#
# https://github.com/deepinsight/insightface
# https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/ncnn/cv/ncnn_glint_arcface.cpp

class ArcFace:
    def __init__( self, num_threads=-1, use_gpu=False, use_lightmode = False):
        self.num_threads = num_threads
        self.use_gpu = use_gpu
        self.lightmode = use_lightmode

        # Network
        # Original Paper: 112x112 or 224x224 Face, 
        # No normalization is applied
        # Region of interest of face from retinaface is usually 96x112 but input to arface is square and 112x112
        
        self.ModelConfig = {
            'height'     : 112,
            'width'      : 112,
            'pad'        : True,
            'name'       : "mobilefacenet/mobilefacenet",
            'mean'       : 0.0,
            'std'        : 1.0,
        }

        self.m = (self.ModelConfig['mean'],self.ModelConfig['mean'],self.ModelConfig['mean'])
        self.s = (1./self.ModelConfig['std'],1./self.ModelConfig['std'],1./self.ModelConfig['std'])

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu
        self.net.opt.num_threads = ncnn.get_big_cpu_count()

        self.net.load_param("./models/" + self.ModelConfig['name'] +".param")
        self.net.load_model("./models/" + self.ModelConfig['name'] +".bin")

    def updateNormalization(self):
        self.m = (self.ModelConfig['mean'],self.ModelConfig['mean'],self.ModelConfig['mean'])
        self.s = (1./self.ModelConfig['std'],1./self.ModelConfig['std'],1./self.ModelConfig['std'])

    def Zscore(self, mat):
        mean, std = cv2.meanStdDev(mat)
        # return ((mat-mean) / std)
        return cv2.divide(cv2.subtract(mat,mean), std)
        
    def __del__(self):
        self.net = None

    def __call__(self, img, Zscore=False):

        tic_start = time.perf_counter()

        (height, width) = img.shape[:2]
        newWidth  = self.ModelConfig['width']
        newHeight = self.ModelConfig['height']
        
        # when ROI is extracted ideally it will be 112x112 and not require scaling but scaling option is provided here
        if height != newHeight or width != newWidth:
            img_rp, factor, l, t = resizeImg(img, newWidth, newHeight, pad=self.ModelConfig['pad'])  
            (height_rp, width_rp) = img_rp.shape[:2]
        else:
            img_rp    = img
            height_rp = newHeight
            width_rp  = newWidth
            factor = 1.0
            l = t = 0

        # Normalization
        #
        # Options are:
        #  a) no normalization ** this is the preferred approach
        #  b) subtract 127.5 then divide 128 gives -1..1 range
        #  c) zscore is (image-mean/std)
        #
        # Approach for scale -1..1: b)
        #  img_arc = np.float32(img_face)
        #  1) img_arc = cv2.divide(cv2.subtract(img,np.array([(127.5,127.5,127.5)])), np.array([(128.,128.,128.)]))
        #  2) img_arc = (img_arc - 127.5)/128.
        #
        # Approach for zscore: c)
        #  img_arc = np.float32(img_face)
        #  mean, std = cv2.meanStdDev(img_arc)    
        #  img_arc = cv2.divide(cv2.subtract(img_arc, mean.T), std.T)
        
        # Arcface operates on RGB images
        mat_in = ncnn.Mat.from_pixels(img_rp, ncnn.Mat.PixelType.PIXEL_BGR2RGB, width_rp, height_rp)

        # Arface has no normalization
        #if self.ModelConfig['mean'] != 0.0 or self.ModelConfig['std'] != 1.0: 
        #    mat_in.substract_mean_normalize(self.m, self.s)

        # Create an extractor
        ex = self.net.create_extractor()
        if not (self.num_threads == -1): ex.set_num_threads(self.num_threads)
        ex.set_light_mode(self.lightmode)
        ex.input("data", mat_in)
        tic_prep = time.perf_counter()

        # Create face features
        mat_out = ex.extract("fc1")
        if mat_out[0] != -100: embeddings = np.array(mat_out[1])
        else:                  embeddings = []

        tic_extract = time.perf_counter()
            
        if Zscore:  self.embeddings = self.Zscore(embeddings)
        tic_zscore = time.perf_counter()

        return embeddings, np.array([ 
            tic_prep-tic_start, 
            tic_extract-tic_prep,
            tic_zscore-tic_extract
        ])
     
if __name__ == '__main__':
    import cv2
    import time
    
    from retinaface import RetinaFace
    from utils_image import resizeImg

    # img = cv2.imread('images/Urs.jpg')
    img = cv2.imread('images/Angelina.jpg')
    # img = cv2.imread('images/Pic_Team1.jpg')
    # img = cv2.imread('images/Pic_Team2.jpg')
    # img = cv2.imread('images/Pic_Team3.jpg')
    # img = cv2.imread('images/worlds-largest-selfie.jpg')
    if img is None: print('Error opening image')

    # Retina Face Finder CNN
    net_retina = RetinaFace(prob_threshold=0.8, nms_threshold=0.4, num_threads=4, use_gpu=False, use_lightmode=False)    
    # Face Features
    net_arc = ArcFace(num_threads=-1, use_gpu=False, use_lightmode=False)

    # Find the faces
    display_img = img.copy()
    faces, face_times = net_retina(img, scale=True, use_weighted_nms=False)
    for face in faces:
        face.draw(display_img)    
    cv2.imshow('Faces Retina', display_img)    
    cv2.waitKey()
        
    # Create embeddings
    for face in faces:
        # We need square bounding box
        face.square()
        # Extractr region of interest
        img_roi, mat_trans, object = extractObjectROI(img, face, target_size = 112, simple=True) # extract ROI
        
        # Debug
        # display_img = img_roi.copy()        
        # object.draw(display_img)        
        # cv2.imshow('Face', display_img)    
        # cv2.waitKey()

        embeddings, arc_times = net_arc(img_roi)
