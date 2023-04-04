import ncnn
import time
import numpy as np
import cv2
import cython
from utils import resizeImg
from utils_cnn import nms, nms_weighted, nms_combination
 
#########################################################################
# FastestDet Class
###########################################################################

@cython.cfunc
def sigmoid_np(x):
    return 1. / (1. + np.exp(-x))

@cython.cfunc
def tanh_np(x):
    return 2.0 / (1.0 + np.exp(-2.*x)) - 1

# https://github.com/dog-qiuqiu/FastestDet
# https://zhuanlan.zhihu.com/p/536500269

class FastestDet():

    def __init__(self, prob_threshold=0.65, nms_threshold=0.45, num_threads=-1, use_gpu=False, use_lightmode=True ):

        self.prob_threshold = prob_threshold
        self.nms_threshold  = nms_threshold
        self.num_threads    = num_threads
        self.use_gpu        = use_gpu
        self.lightmode      = use_lightmode
        
        self.ModelConfig = {
            'height'     : 352,
            'width'      : 352,
            'targetsize' : -1,
            'pad'        : True,
            'base'       : -1,
            'name'       : "fastestdet/FastestDet", 
            'mean'       : 0.,
            'std'        : 255.,
        }
        
        self.m = (  self.ModelConfig['mean'],    self.ModelConfig['mean'],    self.ModelConfig['mean'])
        self.s = (1./self.ModelConfig['std'],  1./self.ModelConfig['std'],  1./self.ModelConfig['std'])

        self.objnet = ncnn.Net()
        self.objnet.clear()
        
        ncnn.set_cpu_powersave(2)
        ncnn.set_omp_num_threads(ncnn.get_big_cpu_count())        

        self.objnet.opt = ncnn.Option()
        self.objnet.opt.use_vulkan_compute = self.use_gpu
        self.objnet.opt.num_threads = ncnn.get_big_cpu_count()
                
        self.objnet.load_param("./models/" + self.ModelConfig['name'] +".param")
        self.objnet.load_model("./models/" + self.ModelConfig['name'] +".bin")
                      
    def __del__(self):
        self.net = None

    def __call__(self, img_bgr, scale=True, use_weighted_nms=True):
        '''
        Extract hands from image
        '''
        tic_start = time.perf_counter()

        (height, width) = img_bgr.shape[:2]
        # Scale image
        if scale:
            newWidth  = self.ModelConfig['width']
            newHeight = self.ModelConfig['height']
        else: 
            # Square image even if we dont scale
            newWidth  = width
            newHeight = height
            
        img_rp, factor, left, top = resizeImg(img_bgr, newWidth, newHeight, pad=self.ModelConfig['pad'])  
        (height_rp, width_rp) = img_rp.shape[:2]

        # convert to ncnn matrix
        mat_in = ncnn.Mat.from_pixels(img_rp, ncnn.Mat.PixelType.PIXEL_BGR, width_rp, height_rp)
        
        # Normalize
        mat_in.substract_mean_normalize(self.m, self.s)

        # Create an extractor
        ex = self.objnet.create_extractor() # every time a new extractor to clear internal caches
        if not (self.num_threads == -1): ex.set_num_threads(self.num_threads)
        ex.set_light_mode(self.lightmode)
        ex.input("input.1", mat_in)
        tic_prep = time.perf_counter()

        # Extract boxes and scores
        tmp = ex.extract("758")
        if tmp[1].w == 0:  return [], np.array([0, 0, 0, 0, 0, 0]) 
        out=tmp[1].numpy()   
        tic_extract = time.perf_counter()

        # Decode boxes
        # c,h,w
        # 85,22,22
        c_out,h_out,w_out = out.shape
        
        # need to figure out organization of out
        # 0: obj score
        # 1: x_offset
        # 2: y_offset
        # 3: width
        # 4: height
        # 5..84: class scores
        
        obj_score  = out[0,:,:]
        cls_score  = out[5:85,:,:]

        category   = np.argmax(cls_score,axis=0) # 168us indx where cls_score is maximum for each column in 22x22 matrix
        max_score  = np.max(cls_score,axis=0)    # 172us max cls_score is faster than take_along_axis
        # max_score  = np.take_along_axis(cls_score, np.expand_dims(category, axis=0), axis=0).squeeze() # 235us, as if calling max
        score      = np.power(max_score, 0.4) * np.power(obj_score, 0.6) # 49us

        picked     = score > self.prob_threshold        # 13us
        p          = score[picked]
        label      = category[picked]
        x_offset   = tanh_np(out[1,picked])
        y_offset   = tanh_np(out[2,picked])
        box_width  = sigmoid_np(out[3,picked])
        box_height = sigmoid_np(out[4,picked])
        
        h,w = np.mgrid[0:w_out,0:h_out]
        cx  = (w[picked] + x_offset) / (w_out)
        cy  = (h[picked] + y_offset) / (h_out)

        # box on resized image
        x0 = ((cx - box_width /2.) * (width_rp) )
        y0 = ((cy - box_height/2.) * (height_rp))
        x1 = ((cx + box_width /2.) * (width_rp) )
        y1 = ((cy + box_height/2.) * (height_rp))
        
        # box on original image
        x0 = (x0 - left) / factor
        x1 = (x1 - left) / factor
        y0 = (y0 - top)  / factor
        y1 = (y1 - top)  / factor

        # make sure its within image
        x0=np.clip(x0, 0., width -1.)
        x1=np.clip(x1, 0., width -1.)
        y0=np.clip(y0, 0., height-1.)
        y1=np.clip(y1, 0., height-1.)
                                
        tic_decode = time.perf_counter()
        
        # Non Maximum Suppression 
        if use_weighted_nms:
            objects = nms_weighted(self.nms_threshold,x0,y0,x1,y1,label,p,use_weighted=use_weighted_nms) # 0.79-0.85ms, 0.72-0.84ms cythonized            
        else:
            objects = nms(self.nms_threshold,x0,y0,x1,y1,label,p) # 0.28ms, 0.14ms cythonized
        # objects = nms_combination(self.nms_threshold,x0,y0,x1,y1,label,p,use_weighted=use_weighted_nms) # 1.49ms, 1.41ms cythonized
        tic_nms = time.perf_counter()
        tic_rotate = time.perf_counter()
        tic_createobject = time.perf_counter()

        return objects, \
               np.array([tic_prep-tic_start, 
                         tic_extract-tic_prep,
                         tic_decode-tic_extract, 
                         tic_nms-tic_decode, 
                         tic_rotate-tic_nms, 
                         tic_createobject-tic_rotate])
        
###########################################################################
# Main Testing
###########################################################################
# Origin: Urs Utzinger

if __name__ == '__main__':
    import cv2
    import time
    from fastestdet import FastestDet
    
    # img = cv2.imread('images/hand.jpg')
    # img = cv2.imread('images/eagle.jpg')
    # img = cv2.imread('images/giraffe.jpg')
    # img = cv2.imread('images/horses.jpg')
    # img = cv2.imread('images/kite.jpg')
    # img = cv2.imread('images/person.jpg')
    # img = cv2.imread('images/scream.jpg')
    img = cv2.imread('images/dog.jpg')
    # img = cv2.imread('images/bus.jpg')
    # img = cv2.imread('images/horseanddog.jpg')
    # img = cv2.imread('images/prom.jpg')
    # img = cv2.imread('images/soccerplayer.jpg')
    # img = cv2.imread('images/zidane.jpg')
    # img = cv2.imread('images/girl-5204299_640.jpg')
    if img is None: print('Error opening image')

    # find faces, low threshold shows performance of algorithm, make it large for real world application
    objectnet = FastestDet(prob_threshold=0.65, nms_threshold=0.45, use_gpu=False)
    
    objects, times_objects = objectnet(img, scale=True, use_weighted_nms=True)

    for object in objects:
        object.draw(img)
    
    cv2.imshow('Objects', img)    
    cv2.waitKey()

    print("Preprocess  {:.2f} ms".format(1000.*times_objects[0]))
    print("Extract     {:.2f} ms".format(1000.*times_objects[1]))
    print("Decode      {:.2f} ms".format(1000.*times_objects[2]))
    print("Select NMS  {:.2f} ms".format(1000.*times_objects[3]))
    print("Rotate      {:.2f} ms".format(1000.*times_objects[4]))
    print("Create Obj  {:.3f} ms".format(1000.*times_objects[5]))

    objectsnet = None
