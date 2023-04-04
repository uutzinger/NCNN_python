###########################################################################
# Retinaface Class
###########################################################################

# https://github.com/Tencent/ncnn/blob/master/python/ncnn/model_zoo/retinaface.py
# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# https://github.com/Qengineering/Face-Recognition-Raspberry-Pi-64-bits/blob/main/src/TRetina.cpp

import numpy as np
import ncnn
import cython
import time
import cv2
from math import sqrt, degrees
from utils_image import resizeImg2Targetsize
from utils_object import objectTypes
from utils_cnn import nms_cv, nms_weighted

class RetinaFace:
    
    def __init__(self, prob_threshold=0.8, nms_threshold=0.4, num_threads=-1, use_gpu=False, use_lightmode=True ):

        self.prob_threshold = prob_threshold
        self.nms_threshold  = nms_threshold
        self.num_threads    = num_threads
        self.use_gpu        = use_gpu
        self.lightmode      = use_lightmode

        # Network Specifications
        self.ModelConfig = {
            'height'     : -1,
            'width'      : -1,
            'targetsize' : 320, # any size works
            'pad'        : False,
            'base'       : -1,
            'name'       : "retina/mnet.25-opt",
            'mean'       : 0.0, # no normalization
            'std'        : 1.0  # no normalization
        }
        
        ncnn.set_cpu_powersave(2)
        ncnn.set_omp_num_threads(ncnn.get_big_cpu_count())
        
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu
        self.net.opt.num_threads = ncnn.get_big_cpu_count()
                
        # Original model is from
        # https://github.com/deepinsight/insightface/
        # https://github.com/deepinsight/insightface/issues/669
        #
        # The ncnn model is from
        # https://github.com/nihui/ncnn-assets/tree/master/models
        # "retinaface-R50.param"
        # "retinaface-R50.bin"
            
        self.net.load_param("./models/" + self.ModelConfig['name'] +".param")
        self.net.load_model("./models/" + self.ModelConfig['name'] +".bin")
        
        base_size   = 16
        ratios = ncnn.Mat(1)
        ratios[0] = 1.0
        scales = ncnn.Mat(2)
        scales[0] = 32.0
        scales[1] = 16.0
        self.anchors32 = self.generate_anchors(base_size, ratios, scales)

        base_size   = 16
        ratios = ncnn.Mat(1)
        ratios[0] = 1.0
        scales = ncnn.Mat(2)
        scales[0] = 8.0
        scales[1] = 4.0
        self.anchors16 = self.generate_anchors(base_size, ratios, scales)

        base_size   = 16
        ratios = ncnn.Mat(1)
        ratios[0] = 1.0
        scales = ncnn.Mat(2)
        scales[0] = 2.0
        scales[1] = 1.0
        self.anchors8 = self.generate_anchors(base_size, ratios, scales)
               
    def __del__(self):
        self.net = None

    def __call__(self, img, scale=True, use_weighted_nms=False):
        # https://github.com/Tencent/ncnn/blob/master/python/ncnn/model_zoo/retinaface.py

        tic_start = time.perf_counter()

        (height, width) = img.shape[:2]
        
        # Stretch factor, stretch the same in horizontal and vertial to maintain aspect ratio
        if scale:
            # scale with aspect ratio maintained
            img_rp, factor, l, t = resizeImg2Targetsize(img, targetsize=self.ModelConfig['targetsize'], base=self.ModelConfig['base'])
            # img_rp, factor, l, t = resizeImg(img, self.ModelConfig['targetsize'], self.ModelConfig['targetsize'], self.ModelConfig['pad'])  
            (height_rp, width_rp) = img_rp.shape[:2]
            mat_in = ncnn.Mat.from_pixels(img_rp, ncnn.Mat.PixelType.PIXEL_BGR2RGB, width_rp, height_rp)
        else:
            # feed original image
            # Convert to ncnn Mat
            mat_in = ncnn.Mat.from_pixels(img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, width, height)
        
        # Normalize
        # no normalize for retinaface
        # mat_in.substract_mean_normalize(self.m, self.s)
        
        ex = self.net.create_extractor() # every time a new extractor to clear internal caches
        if not (self.num_threads == -1): ex.set_num_threads(self.num_threads)
        ex.set_light_mode(self.lightmode)
        ex.input("data", mat_in)
        tic_prep = time.perf_counter()

        (x0_32, y0_32, x1_32, y1_32, prob_32, label_32, kx_32, ky_32, time_extract32, time_proposal32) = self.detect_stride32(ex)
        (x0_16, y0_16, x1_16, y1_16, prob_16, label_16, kx_16, ky_16, time_extract16, time_proposal16) = self.detect_stride16(ex)
        (x0_8,  y0_8,  x1_8,  y1_8,  prob_8,  label_8,  kx_8,  ky_8,  time_extract8,  time_proposal8)  = self.detect_stride8(ex)
        time_extract  = time_extract32  + time_extract16  + time_extract8 
        time_proposal = time_proposal32 + time_proposal16 + time_proposal8

        # Bounding boxes
        x0 = np.concatenate((x0_32, x0_16, x0_8))
        y0 = np.concatenate((y0_32, y0_16, y0_8))
        x1 = np.concatenate((x1_32, x1_16, x1_8))
        y1 = np.concatenate((y1_32, y1_16, y1_8))
        
        # ys = np.vstack([ys, xs]) if ys.size else xs
        # Keypoints
        kx = np.concatenate((kx_32, kx_16, kx_8))
        ky = np.concatenate((ky_32, ky_16, ky_8))
        # Score
        p  = np.concatenate((prob_32, prob_16, prob_8))
        label  = np.concatenate((label_32, label_16, label_8))

        tic_decode = time.perf_counter()

        # Non Maximum Suppression
        #####################################################  
        if use_weighted_nms:
            objects = nms_weighted(
                nms_threshold = self.nms_threshold,
                x0 = x0, y0 = y0,
                x1 = x1, y1 = y1,
                l  = label,
                p  = p,
                kx = kx, ky = ky,
                use_weighted=use_weighted_nms,            
                type=objectTypes['face5'])
        else: 
            objects = nms_cv(
                nms_threshold = self.nms_threshold, 
                x0 = x0, y0 = y0, 
                x1 = x1, y1 = y1, 
                l  = label,
                p  = p,
                kx = kx, ky = ky,
                type=objectTypes['face5']
            )
        tic_nms = time.perf_counter()

        # Update Objects
        faces=[]
        for object in objects:
            # scale object to original image
            if scale: object.resize(scale=factor, l=l, t=t)  
            faces.append(object)

        return faces, np.array([ 
                        tic_prep-tic_start, 
                        tic_decode-tic_prep,
                        tic_nms-tic_decode,
                        time_extract, time_proposal])

    def detect_stride32(self, ex):
        tic_start = time.perf_counter()

        _, score_blob    = ex.extract("face_rpn_cls_prob_reshape_stride32")
        _, bbox_blob     = ex.extract("face_rpn_bbox_pred_stride32")
        _, landmark_blob = ex.extract("face_rpn_landmark_pred_stride32")

        score_blob_np    = score_blob.numpy()
        bbox_blob_np     = bbox_blob.numpy()
        landmark_blob_np = landmark_blob.numpy() 
        
        time_extract = time.perf_counter()

        (x0, y0, x1, y1, prob, label, kx, ky) = self.generate_proposals_np(
            self.anchors32,
            32,
            score_blob_np,
            bbox_blob_np,
            landmark_blob_np,
            self.prob_threshold,
        )
        time_proposal = time.perf_counter()

        return (x0, y0, x1, y1, prob, label, kx, ky, time_extract-tic_start, time_proposal-time_extract)

    def detect_stride16(self, ex):
        tic_start = time.perf_counter()

        _, score_blob    = ex.extract("face_rpn_cls_prob_reshape_stride16")
        _, bbox_blob     = ex.extract("face_rpn_bbox_pred_stride16")
        _, landmark_blob = ex.extract("face_rpn_landmark_pred_stride16")

        score_blob_np    = score_blob.numpy()
        bbox_blob_np     = bbox_blob.numpy()
        landmark_blob_np = landmark_blob.numpy() 
        time_extract = time.perf_counter()

        (x0, y0, x1, y1, prob, label, kx, ky) = self.generate_proposals_np(
            self.anchors16,
            16,
            score_blob_np,
            bbox_blob_np,
            landmark_blob_np,
            self.prob_threshold,
        )
        time_proposal = time.perf_counter()

        return (x0, y0, x1, y1, prob, label, kx, ky, time_extract-tic_start, time_proposal-time_extract)

    def detect_stride8(self, ex):
        tic_start = time.perf_counter()
        
        _, score_blob    = ex.extract("face_rpn_cls_prob_reshape_stride8")
        _, bbox_blob     = ex.extract("face_rpn_bbox_pred_stride8")
        _, landmark_blob = ex.extract("face_rpn_landmark_pred_stride8")

        score_blob_np    = score_blob.numpy()
        bbox_blob_np     = bbox_blob.numpy()
        landmark_blob_np = landmark_blob.numpy() 
        time_extract = time.perf_counter()

        (x0, y0, x1, y1, prob, label, kx, ky) = self.generate_proposals_np(
            self.anchors8,
            8,
            score_blob_np,
            bbox_blob_np,
            landmark_blob_np,
            self.prob_threshold,
        )
        time_proposal = time.perf_counter()

        return (x0, y0, x1, y1, prob, label, kx, ky, time_extract-tic_start, time_proposal-time_extract)

    def generate_anchors(self, base_size:cython.int, ratios, scales):
        # does not need optimization, runs during initialization
        i: cython.int
        j: cython.int
        
        num_ratio = int(ratios.w)
        num_scale = int(scales.w)

        # anchors = ncnn.Mat()
        # anchors.create(w=4, h=num_ratio * num_scale)
        anchors = np.zeros((num_ratio * num_scale, 4), dtype=np.float32)

        cx = base_size * 0.5
        cy = base_size * 0.5

        for i in range(num_ratio):
            
            ar = ratios[i]
            
            r_w = round(base_size / sqrt(ar))
            r_h = round(r_w * ar)  # round(base_size * np.sqrt(ar))

            for j in range(num_scale):
                
                scale = scales[j]

                rs_w = r_w * scale
                rs_h = r_h * scale

                anchor    = anchors[i * num_scale + j]
                # anchor    = anchors.row(i * num_scale + j)
                anchor[0] = cx - rs_w * 0.5
                anchor[1] = cy - rs_h * 0.5
                anchor[2] = cx + rs_w * 0.5
                anchor[3] = cy + rs_h * 0.5
        
        # return ncnn.Mat(anchors)
        return anchors

    def generate_proposals(self, \
            anchors, 
            feat_stride, 
            score_blob, 
            bbox_blob,
            landmark_blob, 
            prob_threshold):

        i: cython.int
        q: cython.int
        
        (h,w) = score_blob.shape[1:3]
        j = np.arange(w)

        # generate face proposal from bbox deltas and shifted anchors
        num_anchors = anchors.shape[0] # anchors num rows

        obj_x0    = np.array([])
        obj_y0    = np.array([])
        obj_x1    = np.array([])
        obj_y1    = np.array([])
        obj_prob  = np.array([])
        obj_label = np.array([])
        obj_kx    = np.empty(shape=[0,5])
        obj_ky    = np.empty(shape=[0,5])
        
        for q in range(num_anchors):
            
            anchor = anchors[q,:]                                         # 1x4

            q4       = q*4
            q10      = q*10
            score    = score_blob[q + num_anchors, :, :]                  # 22x38
            idx      = np.arange(q4,q4+4)
            bbox     = bbox_blob[idx,:,:]                                 # 4x22x38
            idx      = np.arange(q10,q10+10)
            landmark = landmark_blob[idx,:,:]                             # 10x22x38

            # shifted anchor
            anchor_w = anchor[2] - anchor[0]
            anchor_h = anchor[3] - anchor[1]
            
            # should look into vectorizing this axis too
            for i in range(h):

                prob = score[i,j]
                picked = (prob >= prob_threshold)
                
                j_picked = j[picked]

                if len(j_picked) > 0:
                    anchor_x = anchor[0] + (j_picked*feat_stride)
                    anchor_y = np.full(anchor_x.shape, anchor[1] + i*feat_stride)

                    # apply center size
                    dx = bbox[0, i, j_picked]
                    dy = bbox[1, i, j_picked]
                    dw = bbox[2, i, j_picked]
                    dh = bbox[3, i, j_picked]

                    cx = anchor_x + anchor_w * 0.5
                    cy = anchor_y + anchor_h * 0.5

                    pb_cx = cx + anchor_w * dx
                    pb_cy = cy + anchor_h * dy

                    pb_w = anchor_w * np.exp(dw)
                    pb_h = anchor_h * np.exp(dh)

                    x0 = pb_cx - pb_w * 0.5
                    y0 = pb_cy - pb_h * 0.5
                    x1 = pb_cx + pb_w * 0.5
                    y1 = pb_cy + pb_h * 0.5
                    
                    p = prob[j_picked]
                    k_x = (anchor_w + 1) * landmark[0::2,i,j_picked].transpose() + np.expand_dims(cx, axis=1)  
                    k_y = (anchor_h + 1) * landmark[1::2,i,j_picked].transpose() + np.expand_dims(cy, axis=1)

                    obj_x0    = np.concatenate((obj_x0, x0))
                    obj_y0    = np.concatenate((obj_y0, y0))
                    obj_x1    = np.concatenate((obj_x1, x1))
                    obj_y1    = np.concatenate((obj_y1, y1))
                    obj_prob  = np.concatenate((obj_prob, p))
                    obj_label = np.concatenate((obj_label, np.ones(p.shape)))
                    obj_kx    = np.concatenate((obj_kx, k_x))
                    obj_ky    = np.concatenate((obj_ky, k_y))

        return (obj_x0, obj_y0, obj_x1, obj_y1, obj_prob, obj_label, obj_kx, obj_ky)

    def generate_proposals_np(self, \
            anchors, 
            feat_stride, 
            score_blob, 
            bbox_blob,
            landmark_blob, 
            prob_threshold):

        # Optimization
        # q loops over anchors, not parallelized
        # j loops over width of feature map, is parallelized
        # i loops over height of feature map, is parallelied
        q: cython.int
        
        (h,w) = score_blob.shape[1:3]
        i,j   = np.indices((h,w))

        # generate face proposal from bbox deltas and shifted anchors
        num_anchors = anchors.shape[0] # anchors num rows
        
        obj_x0    = np.array([])
        obj_y0    = np.array([])
        obj_x1    = np.array([])
        obj_y1    = np.array([])
        obj_prob  = np.array([])
        obj_label = np.array([])
        obj_kx    = np.empty(shape=[0,5])
        obj_ky    = np.empty(shape=[0,5])

        for q in range(num_anchors):
            
            anchor = anchors[q,:]                        # 1x4

            # shifted anchor
            anchor_w = anchor[2] - anchor[0]
            anchor_h = anchor[3] - anchor[1]

            q4       = q*4
            q10      = q*10
            score    = score_blob[q + num_anchors, :, :] # 22x38
            idx      = np.arange(q4,q4+4)
            bbox     = bbox_blob[idx,:,:]                # 4x22x38
            idx      = np.arange(q10,q10+10)
            landmark = landmark_blob[idx,:,:]            # 10x22x38

            picked   = score >= prob_threshold
            
            if np.sum(picked) > 0:
                i,j = np.indices((h,w))
                j_picked = j[picked]
                i_picked = i[picked]

                dx = bbox[0, i_picked, j_picked]
                dy = bbox[1, i_picked, j_picked]
                dw = bbox[2, i_picked, j_picked]
                dh = bbox[3, i_picked, j_picked]
                
                anchor_x = anchor[0] + (j_picked*feat_stride)
                anchor_y = anchor[1] + (i_picked*feat_stride) # anchor with same shape as anchor_x filled with i*...

                cx = anchor_x + anchor_w * 0.5
                cy = anchor_y + anchor_h * 0.5

                pb_cx = cx + anchor_w * dx
                pb_cy = cy + anchor_h * dy

                pb_w = anchor_w * np.exp(dw)
                pb_h = anchor_h * np.exp(dh)

                x0 = pb_cx - pb_w * 0.5
                y0 = pb_cy - pb_h * 0.5
                x1 = pb_cx + pb_w * 0.5
                y1 = pb_cy + pb_h * 0.5

                prob = score[i_picked,j_picked]
                kx = (anchor_w + 1.) * landmark[0::2,i_picked,j_picked] + cx  
                ky = (anchor_h + 1.) * landmark[1::2,i_picked,j_picked] + cy
            
                label = np.ones(prob.shape)

                obj_x0    = np.concatenate((obj_x0, x0))
                obj_y0    = np.concatenate((obj_y0, y0))
                obj_x1    = np.concatenate((obj_x1, x1))
                obj_y1    = np.concatenate((obj_y1, y1))
                obj_prob  = np.concatenate((obj_prob, prob))
                obj_label = np.concatenate((obj_label, label))
                obj_kx    = np.concatenate((obj_kx, kx.T))
                obj_ky    = np.concatenate((obj_ky, ky.T))

        return (obj_x0, obj_y0, obj_x1, obj_y1, obj_prob, obj_label, obj_kx, obj_ky)
                

###########################################################################
# Main Testing
###########################################################################
# Origin: Urs Utzinger

if __name__ == '__main__':
    import cv2
    import time
    
    # img = cv2.imread('images/Urs.jpg')
    # img = cv2.imread('images/Angelina.jpg')
    # img = cv2.imread('images/Pic_Team1.jpg')
    # img = cv2.imread('images/Pic_Team2.jpg')
    # img = cv2.imread('images/Pic_Team3.jpg')
    img = cv2.imread('images/worlds-largest-selfie.jpg')
    if img is None: print('Error opening image')

    # find faces, low threshold shows performance of algorithm, make it large for real world application
    net = RetinaFace(prob_threshold=0.1, nms_threshold=0.3, use_gpu=False)    
    faces, times = net(img, scale=True, use_weighted_nms=False)
    
    # drawObjects(img, objects)

    for face in faces:
        face.draw(img, prob=False)
        face.rotateBoundingBox()
        face.draw(img, prob=False, color=(0,255,0))

    cv2.imshow('Faces Retina', img)    
    cv2.waitKey()

    print("Preprocess  {:.2f} ms".format(1000.*times[0]))
    print("Decode      {:.2f} ms".format(1000.*times[1]))
    print(" Extract    {:.2f} ms".format(1000.*times[3]))
    print(" Proposal   {:.2f} ms".format(1000.*times[4]))
    print("Select NMS  {:.2f} ms".format(1000.*times[2]))

