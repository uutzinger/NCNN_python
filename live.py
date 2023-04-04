#######################################################################
# Antispoofing
#######################################################################
# From: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing-APK/
import cv2
import ncnn
import numpy as np
import time
from copy import deepcopy
from utils_image import extractObjectROI

#################################################################
# LiveFace
#################################################################

class LiveFace:
    
    def __init__(self, num_threads=4, use_gpu=False, use_lightmode=True ):

        self.num_threads = num_threads
        self.use_gpu = use_gpu
        self.lightmode = use_lightmode

        ncnn.set_cpu_powersave(2)
        ncnn.set_omp_num_threads(ncnn.get_big_cpu_count())        

        # Live detection configs
        ########################
        # origin: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing-APK/blob/master/engine/src/main/assets/live/config.json
        # Model 1
        self.ModelConfig1 = {
            'scale'      : 2.7,
            'height'     : 80,
            'width'      : 80,
            'name'       : "live/model_1",
            'mean'       : 0.0,
            'std'        : 1.0,
            'org_resize' : False
        }
        self.m1 = (  self.ModelConfig1['mean'],    self.ModelConfig1['mean'],    self.ModelConfig1['mean'])
        self.s1 = (1./self.ModelConfig1['std'],  1./self.ModelConfig1['std'],  1./self.ModelConfig1['std'])

        self.net1 = ncnn.Net()
        self.net1.clear()
        self.net1.opt.use_vulkan_compute = self.use_gpu
        self.net1.opt.num_threads = ncnn.get_big_cpu_count()

        # Model 2
        self.ModelConfig2 = {
            'scale'      : 4.0,
            'height'     : 80,
            'width'      : 80,
            'mean'       : 0.0,
            'std'        : 1.0,
            'name'       : "live/model_2",
            'org_resize' : False
        }
        self.m2 = (  self.ModelConfig2['mean'],    self.ModelConfig2['mean'],    self.ModelConfig2['mean'])
        self.s2 = (1./self.ModelConfig2['std'],  1./self.ModelConfig2['std'],  1./self.ModelConfig2['std'])
        self.net2 = ncnn.Net()
        self.net2.clear()
        self.net2.opt.use_vulkan_compute = self.use_gpu
        self.net2.opt.num_threads = ncnn.get_big_cpu_count()

        # model is from
        # https://github.com/minivision-ai/Silent-Face-Anti-Spoofing-APK/tree/master/engine/src/main/assets/live
        self.net1.load_param("./models/" + self.ModelConfig1['name'] +".param")
        self.net1.load_model("./models/" + self.ModelConfig1['name'] +".bin")
        self.net2.load_param("./models/" + self.ModelConfig2['name'] +".param")
        self.net2.load_model("./models/" + self.ModelConfig2['name'] +".bin")
        
    def __del__(self):
        self.net1 = None
        self.net2 = None

    def __call__(self, img, face) :
        # From: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing-APK/tree/master/engine/src/main/cpp/live

        tic_start1 = time.perf_counter()

        (img_h, img_w) = img.shape[:2]
        
        (box_w, box_h) = face.width_height()
        max_scale = min( img_w / box_w, img_h / box_h)
        
        face_1 = deepcopy(face)
        face_2 = deepcopy(face)
        
        # Model 1
        ####################################
        scale = min(self.ModelConfig1['scale'], max_scale)
        face_1.square(scale = scale)

        img_Live1, _, _   = extractObjectROI(img, face_1, target_size=self.ModelConfig1['width'], simple=True)
        (height1, width1) = img_Live1.shape[:2]
        # Convert to ncnn mat
        mat_in = ncnn.Mat.from_pixels(img_Live1, ncnn.Mat.PixelType.PIXEL_BGR, width1, height1)

        # Normalize (no normalization)
        # mat_in.substract_mean_normalize(self.m1, self.s1)
        
        # Create extractor
        ex = self.net1.create_extractor()
        if not (self.num_threads==-1): ex.set_num_threads(self.num_threads)
        ex.set_light_mode(self.lightmode)
        ex.input("data", mat_in)

        tic_prep1 = time.perf_counter()
        
        # Extract confidence 1
        mat_out = ex.extract("softmax")
        confidence_1 = mat_out[1][1]

        tic_confidence1 = time.perf_counter()

        # Model 2
        ####################################
        tic_start2 = time.perf_counter()

        scale = min(self.ModelConfig2['scale'], max_scale)
        face_2.square(scale = scale)
        
        img_Live2, _, _   = extractObjectROI(img, face_2, target_size=self.ModelConfig2['width'], simple=True)
        (height2, width2) = img_Live2.shape[:2]
        
        # Convert to ncnn mat
        mat_in = ncnn.Mat.from_pixels(img_Live2, ncnn.Mat.PixelType.PIXEL_BGR, width2, height2)

        # Normalize (no normalization)
        # mat_in.substract_mean_normalize(self.m2, self.s2)
        
        # Create extractor
        ex = self.net2.create_extractor()
        if not (self.num_threads == -1): ex.set_num_threads(self.num_threads)
        ex.set_light_mode(self.lightmode)
        ex.input("data", mat_in);

        tic_prep2 = time.perf_counter()
        
        # Extract confidence 2
        mat_out = ex.extract("softmax")
        # need to debug this structure
        # (0, [0.0404271, 0.0352502, 0.924323])
        confidence_2 = mat_out[1][1]

        tic_confidence2 = time.perf_counter()
            
        return ((confidence_1+confidence_2) / 2.,  np.array([ 
                        tic_prep1-tic_start1, 
                        tic_confidence1-tic_prep1,
                        tic_prep2-tic_start2, 
                        tic_confidence2-tic_prep2]), face_1, face_2)

if __name__ == '__main__':
    import cv2
    import time
    from retinaface import RetinaFace
    
    use_still = True

    # Retina Face Finder CNN
    net_retina = RetinaFace(prob_threshold=0.8, nms_threshold=0.4, num_threads=4, use_gpu=False, use_lightmode=False)

    # Spoofing CNN
    net_live = LiveFace(num_threads=4, use_gpu=False, use_lightmode=False)
   
    times_live = np.zeros(4)
    times_count = 0
    time_cnn = 0.
    
    # img = cv2.imread('images/Urs.jpg')
    # img = cv2.imread('images/Angelina.jpg')
    img = cv2.imread('images/Pic_Team1.jpg')
    # img = cv2.imread('images/Pic_Team2.jpg')
    # img = cv2.imread('images/Pic_Team3.jpg')
    # img = cv2.imread('images/worlds-largest-selfie.jpg')
    if img is None: print('Error opening image')
    m,n = img.shape[:2]
    img = img[0:int(m/2), int(n/4):int(n*3/4), :]

    # (height, width) = img.shape[:2]
    # scale = 0.5
    # h = int(height* scale)
    # w = int(width * scale)
    # img = cv2.resize(img, (w, h), cv2.INTER_LINEAR)

    faces, _ = net_retina(img, scale=True)    

    for obj in faces:  
        tic = time.perf_counter()
        obj.rotateBoundingBox()
        live, times, obj1, obj2  = net_live(img, obj)
        times_live  += times
        times_count += 1
        toc = time.perf_counter()
        time_cnn += toc-tic
        obj.draw(img, color=(0,255,0))
        obj1.drawRect(img, color=(0,255,0))
        obj2.drawRect(img, color=(0,0,255))
        # obj.drawRect(img, color=(0,255,0))
        obj.printText(img, "{:.2f}".format(live))

    cv2.imshow('Live', img)    
    cv2.waitKey()

    if times_count > 0:
        print("Preprocess 1 {:.2f} ms".format(1000.*times_live[0]/times_count))
        print("Extract    1 {:.2f} ms".format(1000.*times_live[1]/times_count))
        print("Preprocess 2 {:.2f} ms".format(1000.*times_live[2]/times_count))
        print("Extract    2 {:.2f} ms".format(1000.*times_live[3]/times_count))
        print("Total        {:.2f} ms".format(1000.*time_cnn/times_count))

    del net_retina
    del net_live
