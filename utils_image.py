########################################################################################
# Image Handling
########################################################################################

import numpy as np
import cv2
import cython
from copy import deepcopy
from utils_object import objectTypes
 
def resizeImg(img, newWidth:cython.int, newHeight:cython.int, pad=False):
    """
    Scale & pad image 
      image will not change aspect ratio
      input: image, width and height
      if padding enabled new image will become requested size
      if padding disabled one dimension of new image might be smaller than requested
      padding mode: uses cv2.BORDER_CONSTANT and add 0/black to pad
      for padding, image is centered and padded top&bottom or left&right
    This function allocates a new image.
    It return the new image, the scaling, left and top shift due to padding
    """
    # origin: UU

    scale: cython.double
    height: cython.int
    width:  cython.int
    h:      cython.int
    w:      cython.int
    l:      cython.int
    r:      cython.int
    t:      cython.int
    b:      cython.int
    wpad:   cython.int
    hpad:   cython.int
        
    (height, width) = img.shape[:2]
    
    if height == 0 or width == 0:
        return img, 1., 0, 0

    # Stretch factor, stretch the same in horizontal and vertial to maintain aspect ratio
    if width/newWidth > height/newHeight:
        # scale width to newWidth
        scale = float(newWidth) / width
        w = newWidth
        h = round(height * scale)
    else:
        # scale height to newHeight
        scale = float(newHeight) / height
        h = newHeight
        w = round(width * scale)
                
    # Resize image
    img_r = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
        
    # Pad image if requested
    if pad:
        wpad = newWidth  - w
        hpad = newHeight - h

        # Center the image and pad
        l = wpad // 2
        t = hpad // 2
        r = wpad - l
        b = hpad - t
    
        if img_r.ndim == 3: size = (h+t+b, w+l+r, img_r.shape[2])
        else:               size = (h+t+b, w+l+r)
        img_rp = np.empty(size, dtype=img_r.dtype) # allocate new image
        cv2.copyMakeBorder(dst=img_rp, src=img_r, top=t, bottom=b, left=l, right=r, borderType=cv2.BORDER_CONSTANT, value=0)
        return img_rp, scale, l, t
    else:
        return img_r,  scale, 0, 0

def resizeImg2Targetsize(img, targetsize:cython.int, base:cython.int = -1):
    """
    Resize & pad image so that largest dimension is targetsize.
    Aspect ratio is preserved.
    Pad image to multiple of base.
    Only one dimension of image will match targetsize.
    This function allocates a new image.
    """
        
    (height, width) = img.shape[:2]
    
    if width > height:
        scale = float(targetsize) / width
        w = targetsize
        h = int(float(height) * scale)
    else:
        scale = float(targetsize) / height
        h = targetsize
        w = int(width * scale)
    
    # Resize image
    img_r = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
        
    if base > 0:
        wpad = int((w+base-1) / base) * base - w
        hpad = int((h+base-1) / base) * base - h

        # Center the image and pad
        l = wpad // 2
        t = hpad // 2
        r = wpad - l
        b = hpad - t
    
        if img_r.ndim == 3: size = (h+t+b, w+l+r, img_r.shape[2])
        else:               size = (h+t+b, w+l+r)
        img_rp = np.empty(size, dtype=img_r.dtype) # allocate new image
        cv2.copyMakeBorder(dst=img_rp, src=img_r, top=t, bottom=b, left=l, right=r, borderType=cv2.BORDER_CONSTANT, value=0)
        return img_rp, scale, l, t
    else:
        return img_r,  scale, 0, 0


def extractObjectROI(img_bgr, obj, target_size, simple=True):
    ''' 
    Produces region of interest image, updated object and transformation matrix to revert coordinates onto original image
    
    Simple: estimates transformation based on location of boundingbox
    Complex: Computes affine transformation matrix based on keypoints compared to average keypoints (face only)
    
    Returns image is affine transfomration or ROI taken from original image, image will be square
    Urs Utzinger, 2023
    '''
    
    # objectTypes = {'rect':0, 'yolo80':1, 'hand':2, 'hand7':3, 'hand21':4, 'face':5, 
    #                'face5':6, 'person':7, 'person4':8, 'person17':9, 'person39':10 }

    obj_out = deepcopy(obj)

    if simple:
        
        # RotatedObject
        if obj.isRotated():
            # Bounding Box is: top left, top right, bottom right, bottom left
            # What is the width and height of the bounding box?
            w = cv2.norm(obj.bb[1,:,:] - obj.bb[0,:,:]) # top right - top left
            h = cv2.norm(obj.bb[2,:,:] - obj.bb[1,:,:]) # bottom right - top right
            srcPts = obj.bb[0:3,:,:]
            if w > h: 
                scale = target_size / w
                new_h = h * scale
                new_w = target_size
            else:
                scale = target_size / h
                new_w = w * scale
                new_h = target_size                    
            # mapping corner points on the target image           
            dstPts = np.array( [
                [ [ 0,         0] ], # top left
                [ [ new_w,     0] ], # top right
                [ [ new_w, new_h] ]  # bottom right
            ], dtype=np.float32)
            trans_mat  = cv2.getAffineTransform(srcPts, dstPts)
            img_affine = cv2.warpAffine(src=img_bgr, M=trans_mat, dsize=(round(new_w), round(new_h)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            obj_out.transform(trans_mat)
            return img_affine, trans_mat, obj_out
        
        # Object has rectangular bounding box
        else:
            # We dont have keypoints we extract ROI and resize it to target size
            # We extract ROI and then resize it to target size
            # We have uniform scaling by f and translation by l,t
            # M    = [ [f,     0,  l]
            #          [0,     f,  t]
            #          [0,     0,  1] ]
            #
            # M^-1 = [ [1/f,   0, -l/f]
            #          [0,   1/f, -t/f]
            #          [0,     0,  1] ]
            #
            
            # Exgtract ROI
            img_roi, obj_roi, factor, l, t = extractRectROI(img=img_bgr, object=obj, pad=False)
            # Transformation Matrix from image to ROI
            trans_mat_roi = np.array( [
                [factor, 0, l],
                [0, factor, t],
                [0,      0, 1]
            ], dtype=np.float32)
            # Resize ROI to target size
            w,h = img_roi.shape[:2]
            if w > h: 
                scale = target_size / w
                new_h = round(h * scale)
                new_w = target_size
            else:
                scale = target_size / h
                new_w = round(w * scale)
                new_h = target_size                                
            img_roi, factor, l, t = resizeImg(img_roi, new_w, new_h, pad=True)
            # Transformation Matrix from ROI to target size
            trans_mat_s   = np.array( [
                [factor,  0, l],
                [0,  factor, t],
                [0,       0, 1]
            ], dtype=np.float32)
            trans_mat = np.matmul(trans_mat_s,trans_mat_roi) 
            obj_out.transform(trans_mat)
            return img_roi, trans_mat, obj_out
        
    elif obj.type == objectTypes['face5']:
        # perspective transformation to 96x112
        # https://github.com/XinghaoChen9/LiveFaceReco_RaspberryPi/blob/master/src/livefacereco.cpp
        
        dstPts = np.array([
            [ [30.2946, 51.6963] ], # eye
            [ [65.5318, 51.5014] ], # eye
            [ [48.0252, 71.7366] ], # nose
            [ [33.5493, 92.3655] ], # mouth
            [ [62.7299, 92.2041] ]  # mouth
        ], dtype=np.float32)

        srcPts = obj.k

        # Compute the transformation matrix
        #
        # Option 1: Umeyama algorithm (slow)
        # trans_mat = align(src_pts, dst_pts, estimate_scale=True)
        # trans_mat = trans_mat[:2,:]
        #
        # Option 2: cv2.estimateAffinePartial2D for scaling, rotation, translation
        # This gibes same results as option 1 but is much faster
        trans_mat, inliers = cv2.estimateAffinePartial2D(srcPts, dstPts, method=cv2.RANSAC, ransacReprojThreshold=5, maxIters=2000, confidence=0.99, refineIters=10)
        #
        # Option 3: cv2.estimateAffine2D for scaling, rotation, shear, translation
        # trans_mat, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5, maxIters=2000, confidence=0.99, refineIters=10)

        # Lets figure out where center of bounding box is after transformation.
        src_pt = obj.center()
        dst_pt = cv2.transform(src_pt,trans_mat)  
        # What is the offset? left and top
        l = target_size/2 - dst_pt[0,0,0]
        t = target_size/2 - dst_pt[0,0,1]
        # adjust the transformation matrix for the shift        
        trans_mat[0,2] += l
        trans_mat[1,2] += t
        
        # inverse transformation matrix 
        trans_mat_inv = cv2.invertAffineTransform(trans_mat)
        # bounding box of the target image
        dst_pts = np.array( [
            [ [            0,             0] ], # top left
            [ [  target_size,             0] ], # top right
            [ [  target_size,   target_size] ], # bottom right
            [ [            0,   target_size] ], # bottom left
            [ [target_size/2, target_size/2] ]  # center
        ], dtype=np.float32)
        # mapping the bounding box back to the source image      
        obj_out.bb = cv2.transform(dst_pts,trans_mat_inv) 

        # the keypoints dont need rotation
        # no need to transform them, they were copied into obj_out
        img_affine  = cv2.warpAffine(src=img_bgr, M=trans_mat, dsize=(target_size, target_size), flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=0) 

    return img_affine, trans_mat, obj_out

def extractRectROI(img, object, pad=True):
    """
    Returns a new image and new object
        Input: Image and object containing rectangle
        Output: ROI and object with coordinates matching ROI
        If padding is enabled, ROI will be the full requested size0, 
    Without padding, ROI might be clipped
    This allocates new image and object
    Urs Utzinger, 2023
    """

    if object.isRotated(): return [], [], 1.0, 0, 0
    
    (height, width) = img.shape[:2]

    object_out = deepcopy(object) # create new object
    
    # indices to the image array
    bb = np.int32(object_out.bb)
    x_start  = bb[0,0,0]
    y_start  = bb[0,0,1]
    x_end    = bb[1,0,0]
    y_end    = bb[1,0,1]
    x_offset = x_start
    y_offset = y_start
    l = 0
    t = 0
        
    if pad:  # check bounds, pad if necessary
        needPadding = False    
        # check bounds and where padding is necessary
        # Left
        if x_start < 0:     
            needPadding = True  
            l = -x_start      
            x_start = 0      
            x_offset -= l 
        # Top
        if y_start < 0:      
            needPadding = True  
            t = -y_start      
            y_start = 0      
            y_offset -= t
        # Right
        if x_end   > width:  
            needPadding = True # x_offset remains
            r = x_end-width   
            x_end   = width  
        else:                
            r = 0
        # Bottom
        if y_end   > height: 
            needPadding = True # y_offset remains
            b = y_end-height  
            y_end   = height 
        else:               
            b = 0

        # create new image
        if img.ndim == 3: 
            size = (y_end-y_start+t+b, x_end-x_start+l+r, img.shape[2])
        else:             
            size = (y_end-y_start+t+b, x_end-x_start+l+r)
        img_out = np.empty(size,dtype=img.dtype) # allocate new image
        if needPadding:
            cv2.copyMakeBorder(dst = img_out, src=img[y_start:y_end, x_start:x_end, :], top=t, bottom=b, left=l, right=r, borderType=cv2.BORDER_CONSTANT, value=0)
        else:
            np.copyto(dst=img_out, src=img[y_start:y_end, x_start:x_end, :])
        
    else: # check bounds, no padding
        if x_start < 0:      x_start = 0
        if y_start < 0:      y_start = 0
        if x_end   > width:  x_end   = width  
        if y_end   > height: y_end   = height 

        if img.ndim == 3: 
            size = (y_end-y_start, x_end-x_start, img.shape[2])
        else:             
            size = (y_end-y_start, x_end-x_start)        
        img_out = np.empty(size,dtype=img.dtype)
        np.copyto(dst=img_out, src=img[y_start:y_end, x_start:x_end, :])

    # correct object locations        
    object_out.bb[:,:,0] -= x_offset
    object_out.bb[:,:,1] -= y_offset
    object_out.k[:,:,0]  -= x_offset
    object_out.k[:,:,1]  -= y_offset
                                
    return img_out, object_out, 1.0, l, t
