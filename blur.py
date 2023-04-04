import numpy as np
import cv2
import time
from copy import deepcopy
from math import floor, sqrt
from utils_image import extractObjectROI

class Blur:

    def __init__( self, c: float=1./8.):
        """
        Input the fraction of spectrum considered high frequency.
        Sets up average face feature locations.
        """         
        self.c  = c
        self.target_size = 128
        
    def __del__(self):
        pass
    
    def __call__(self, img, object, fft: bool = False):
        """
        Estimates the blur in the object region of the image.
        Either use discrete Fourier transform to compute spectrum or use gaussian filter in spatial domain.
        Spatial domain computation is faster.
        Result is power in low and high pass region. 
        Numbers computed from Fourier transfomr and gaussian filter are not the same.
        You will need to experimentally determine when ratio between the two frequency bands is below threshold to be considered blurred. 
        """
        
        region = deepcopy(object)
        region.square()
        target_size = cv2.getOptimalDFTSize(self.target_size)   # optimized image size for faster FFT
        roi, _, _ = extractObjectROI(img, region, target_size=target_size, simple=True)
        (height, width) = roi.shape[:2]        
        roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY) # work with gray scale image and 0..1 scale
        roi = cv2.normalize(roi, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # work with floats 0..1

        # cv2.imshow('before', roi)
        # cv2.waitKey(1)
        
        if fft:
            # Discrete Fourier Transform
            #  frequency range mask, will not use fftshift, directly extract from 4 quadrants
            #  mask == True is high frequency area
            filter_x = filter_y = min(floor(width  * (1.-self.c) / 2.), floor(height * (1.-self.c) / 2.))
            mask=np.full((roi.shape), True)                      # initialize
            mask[        0:filter_y,         0:filter_x] = False # top left
            mask[-filter_y:,                 0:filter_x] = False # top right
            mask[        0:filter_y, -filter_x:        ] = False # bottom left
            mask[-filter_y:,         -filter_x:        ] = False # bottom right
            f     = cv2.dft(roi,flags = cv2.DFT_SCALE | cv2.DFT_COMPLEX_OUTPUT) # Fourier transform
            fmag  = cv2.magnitude(f[:,:,0],f[:,:,1]) # extract magnitude from Fourier spectrum
            pLP   = np.sum(fmag, where=(~mask))      # power/sum in low frequencies
            pHP   = np.sum(fmag, where=mask)         # power/sum in high frequencies
            # pRel   = 20.*np.log10(pHP/(pLP+pHP))   # relative power
        
            # # Debug: Generate the high pass filtered image
            # #  as no windowing has been applied this will generate ripples in the image
            # f_hp = f.copy()
            # f_hp[~mask,:] = 0
            # img_hp = cv2.dft(f_hp, flags=cv2.DFT_INVERSE | cv2.DFT_REAL_OUTPUT)
            # # Generate the low pass filtered image
            # f_lp = f.copy()
            # f_lp[mask,:] = 0
            # img_lp = cv2.dft(f_lp, flags=cv2.DFT_INVERSE | cv2.DFT_REAL_OUTPUT)
            # cv2.imshow('mask', mask*1.0)
            # cv2.imshow('hp f', img_hp*10.)
            # cv2.imshow('lp f', img_lp)
            # cv2.imshow('DFT', fmag*1000.)
            # cv2.waitKey()

            return pHP, pLP

        else:
            # Spatial domain low pass filter
            filter_x = filter_y = max(min(floor(width  * (self.c) / 64.), floor(height * (self.c) / 64.)), 1.0)
            img_lp = cv2.GaussianBlur(roi, ksize=[0,0], sigmaX=filter_x, sigmaY=filter_y, borderType=cv2.BORDER_REFLECT) # low pass
            img_hp = roi - img_lp # high passed image
            m   = cv2.mean(cv2.multiply(img_lp, img_lp)) # power in lowpassed image
            pLP = sqrt(m[0]) # root mean squared power
            m   = cv2.mean(cv2.multiply(img_hp, img_hp)) # power in highpassed image
            pHP = sqrt(m[0]) # root mean squared power
            # pRel   = 20.*np.log10(pHP/(pHP+pLP)     # relative power
            
            # Debug: show higpassed and lowpassed images
            # cv2.imshow('hp s', img_hp*10.)
            # cv2.imshow('lp s', img_lp)
            # cv2.waitKey()
                
            return pHP, pLP
    
###########################################################################
# Main Testing
###########################################################################
# Origin: Urs Utzinger

# C   Method      Power time
# 3/4 time domain 6-7% 10ms,  frequency domain 46-47% 11ms
# 1/2 time domain 5-6% 10ms,  frequency domain 24% 12ms
# 1/4 time domain   3% 10ms,  frequency domain 12% 11ms

if __name__ == '__main__':

    import numpy as np
    import cv2
    import time
    from retinaface import RetinaFace
   
    # find faces
    net = RetinaFace(prob_threshold=0.8, nms_threshold=0.4, num_threads=4, use_gpu=False, use_lightmode=False)

    # image blur    
    blur = Blur(3./4.)

    # Test on exmaple images
    
    # img = cv2.imread('images/Urs.jpg')
    img = cv2.imread('images/Angelina.jpg')
    # img = cv2.imread('images/Pic_Team1.jpg')
    # img = cv2.imread('images/Pic_Team2.jpg')
    # img = cv2.imread('images/Pic_Team3.jpg')
    # img = cv2.imread('images/worlds-largest-selfie.jpg')
    if img is None: print('Error opening image')
    
    faces, _ = net(img, scale=True)

    for obj in faces:
        
        tic = time.perf_counter()
        pHPfft, pLPfft  = blur(img, obj, fft=True)
        # pHPfft, pLPfft , img_ffthp, img_fftlp  = blur(frame, obj, fft=True)
        toc = time.perf_counter()
        print('Frequency Domain: Power HP {:.2f}, Power LP {:.2f}, {:.4f} [seconds]'.format(pHPfft, pLPfft, toc-tic))
        
        tic = time.perf_counter()
        pHPrms, pLPrms = blur(img, obj, fft=False)
        # pHPrms, pLPrms, img_rmshp, img_rmslp = blur(frame, obj, fft=False)
        toc = time.perf_counter()
        print('Spatial Domain: rms HP {:.2f}, rms LP {:.2f}, {:.4f}[seconds]'.format(pHPrms, pLPrms, toc-tic))


    # Test on simulated blurred images
    # Create blurred images with 48 different blur levels
    # and calculate the power in the high and low frequencies using frequency domain and spatial domain
    
    powerfftHP=np.zeros(48)
    powerfftLP=np.zeros(48)
    powerrmsHP=np.zeros(48)
    powerrmsLP=np.zeros(48)
    for i in range(48):
        I_blurr  = cv2.GaussianBlur(img, ksize=[0,0], sigmaX=i+.5, sigmaY=i+.5, borderType=cv2.BORDER_REFLECT) # low pass
        cv2.imshow('blurred', I_blurr)
        cv2.waitKey(1)
        # c=input('Continue')    
        powerfftHP[i], powerfftLP[i] = blur(I_blurr,obj,fft=True)
        powerrmsHP[i], powerrmsLP[i] = blur(I_blurr,obj,fft=False)
    print('Power HP Frequency Domain:')
    np.set_printoptions(precision=4)
    print((powerfftHP/(powerfftHP+powerfftLP)))
    print('Power HP Spatial Domain:')
    print((powerrmsHP/(powerrmsHP+powerrmsLP)))
