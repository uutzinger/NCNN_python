###########################################################
# CNN utility functions
# Urs Utzinger, 2023
###########################################################
import numpy as np
import cv2
import cython
from utils_object import Object, objectTypes
from math import sqrt, ceil

###########################################################
# Non Maximum Supression
###########################################################

def nms_cv(nms_threshold, x0, y0, x1, y1, l, p, kx=[], ky=[], kz=[], v=[], type=objectTypes['rect']):
    # cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh) needs:
    # boxes = [ [x, y, w, h], ...] with x, y, w, h of type int

    if len(p) == 0: return []

    with_keypoints   = True if (len(kx)>0) else False
    with_keypoints2D = True if (len(ky)>0) else False
    with_keypoints3D = True if (len(kz)>0) else False
    with_visibility  = True if (len(v) >0) else False

    objects = []

    categories = np.unique(l)

    for category in categories:
        # select cordinates of the current category boxes
        picked = (l == category)

        x0_c = x0[picked]
        x1_c = x1[picked]
        y0_c = y0[picked]
        y1_c = y1[picked]
        l_c  =  l[picked]
        p_c  =  p[picked]

        if with_keypoints:    kx_c = kx[picked,:]
        if with_keypoints2D:  ky_c = ky[picked,:]
        if with_keypoints3D:  kz_c = kz[picked,:]
        if with_visibility:   v_c  =  v[picked,:] 

        # OpenCV NMS
        boxes_c = np.array([(x0_c+x1_c)/2,(y0_c+y1_c)/2,(x1_c-x0_c),(y1_c-y0_c)]).T
        idx = cv2.dnn.NMSBoxes(boxes_c, p_c, 0, nms_threshold)

        for i in idx:
            # convert and aggregate
            if with_keypoints3D:
                k_c = np.array( [
                    [kx_c[i,:]],
                    [ky_c[i,:]],
                    [kz_c[i,:]]
                    ], dtype=np.float32).T
            elif with_keypoints2D:
                k_c = np.array( [
                    [kx_c[i,:]],
                    [ky_c[i,:]]
                    ], dtype=np.float32).T
            elif with_keypoints:
                k_c = np.array( [
                    [kx_c[i,:]]
                    ], dtype=np.float32).T
            else:
                k_c =[]
            
            obj = Object(\
                        bb = np.array([
                            [[x0_c[i],y0_c[i]]],
                            [[x1_c[i],y1_c[i]]]
                        ], dtype=np.float32),
                        l  =  l_c[i], 
                        p  =  p_c[i],
                        k  =  k_c,
                        v  =  v_c[i,:] if with_visibility else [],
                        type = type)
            objects.append(obj)
        
    return objects

def nms(nms_threshold, x0, y0, x1, y1, l, p, kx=[], ky=[], kz=[], v=[], type=objectTypes['rect']):
    '''
    Simple NMS, non weighting, non numpy accelerated, fast when cythonized and few bounding boxes
    Returns objects
    '''
    i: cython.int
    j: cython.int
    
    if len(p) == 0: return []

    with_keypoints   = True if (len(kx)>0) else False
    with_keypoints2D = True if (len(ky)>0) else False
    with_keypoints3D = True if (len(kz)>0) else False
    with_visibility  = True if (len(v) >0) else False
    
    picked  = []
    objects = []
    idx = np.argsort(p)[::-1]
    
    x0_s = x0[idx]
    y0_s = y0[idx]
    x1_s = x1[idx]
    y1_s = y1[idx]
    l_s  =  l[idx]
    p_s  =  p[idx]
    
    if with_keypoints:    kx_s = kx[idx,:]
    if with_keypoints2D:  ky_s = ky[idx,:]
    if with_keypoints3D:  kz_s = kz[idx,:]
    if with_visibility:   v_s  =  v[idx,:] 
            
    src_boxes_area = (x1_s - x0_s + 1) * (y1_s - y0_s + 1)
    
    for i in range(len(p_s)):
        keep = True
        for j in range(len(picked)):
            
            if x0_s[i]>x1_s[picked[j]] or x1_s[i]<x0_s[picked[j]] or y0_s[i]>y1_s[picked[j]] or y1_s[i]<y0_s[picked[j]]:
                inter_area = 0
            else:
                x0_i = max(x0_s[i], x0_s[picked[j]])
                y0_i = max(y0_s[i], y0_s[picked[j]])
                x1_i = min(x1_s[i], x1_s[picked[j]])
                y1_i = min(y1_s[i], y1_s[picked[j]])
                w_i  = x1_i - x0_i + 1
                h_i  = y1_i - y0_i + 1        
                # overlapping area
                inter_area = w_i * h_i

            union_area = src_boxes_area[i] + src_boxes_area[picked[j]] - inter_area
            IoU = inter_area / union_area
            
            if (IoU > nms_threshold) and (l_s[i] == l_s[picked[j]]): 
                keep = False
                break
        if keep:
            picked.append(i)
            
    # aggregate detections into objects
    for j in range(len(picked)):
        idx = picked[j]
        if with_keypoints3D:
            k_s = np.array( [
                [kx_s[idx,:]],
                [ky_s[idx,:]],
                [kz_s[idx,:]]
                ], dtype=np.float32).T
        elif with_keypoints2D:
            k_s = np.array( [
                [kx_s[idx,:]],
                [ky_s[idx,:]]
                ], dtype=np.float32).T
        elif with_keypoints:
            k_s = np.array( [
                [kx_s[idx,:]]
                ], dtype=np.float32).T
        else:
            k_s =[]
        obj = Object(\
                     bb = np.array( [
                        [[x0_s[idx],y0_s[idx]]],
                        [[x1_s[idx],y1_s[idx]]]                         
                     ], dtype=np.float32),
                     l  =  l_s[idx], 
                     p  =  p_s[idx],
                     k  =  k_s,
                     v  =  v_s[idx,:] if with_visibility else [],
                     type = type)
        objects.append(obj)
        
    return objects


def nms_combination(nms_threshold, x0, y0, x1, y1, l, p, kx=[], ky=[], kz=[], v=[], type=objectTypes['rect'], use_weighted=True):
    ''' 
    NMS
     computes area of all boxes
     creates list of pairwise combinations of boxes
     computes area of all intersections
     merges if threshold met
    numpy accelerated
    Returns objects
    '''
    
    i: cython.int

    if len(p) == 0: return []

    with_keypoints   = True if (len(kx)>0) else False
    with_keypoints2D = True if (len(ky)>0) else False
    with_keypoints3D = True if (len(kz)>0) else False
    with_visibility  = True if (len(v) >0) else False
     
    objects = []
    categories = np.unique(l)
    idx = np.argsort(p)[::-1]
    x0 = x0[idx]
    x1 = x1[idx]
    y0 = y0[idx]
    y1 = y1[idx]
    p  = p[idx]
    l  = l[idx]
    if with_keypoints:    kx = kx[idx,:]
    if with_keypoints2D:  ky = ky[idx,:]
    if with_keypoints3D:  kz = kz[idx,:]
    if with_visibility:   v  =  v[idx,:] 
    
    for category in categories:
        # select cordinates of the current category boxes
        picked = (l == category)
        x0_cs = x0[picked]
        x1_cs = x1[picked]
        y0_cs = y0[picked]
        y1_cs = y1[picked]
        p_cs  =  p[picked]
        if with_keypoints:    kx_cs = kx[picked,:]
        if with_keypoints2D:  ky_cs = ky[picked,:]
        if with_keypoints3D:  kz_cs = kz[picked,:]
        if with_visibility:   v_cs  =  v[picked,:] 

        # area of all rectangles
        w_cs  = x1_cs - x0_cs + 1 # width
        h_cs  = y1_cs - y0_cs + 1 # height
        a_cs  = w_cs * h_cs       # area

        # compare each box to all other boxes without repetitions
        # use itertools.combinations equivalent for numpy         
        num_rects = len(a_cs)
        nrr = np.arange(num_rects)
        idx = np.stack(np.triu_indices(len(nrr), k=0), axis=-1)

        # calculate intersection area of two rectangles   
        x0_o = np.maximum(x0_cs[idx[:,0]], x0_cs[idx[:,1]])
        y0_o = np.maximum(y0_cs[idx[:,0]], y0_cs[idx[:,1]])
        x1_o = np.minimum(x1_cs[idx[:,0]], x1_cs[idx[:,1]])
        y1_o = np.minimum(y1_cs[idx[:,0]], y1_cs[idx[:,1]])
        w_o  = np.maximum((x1_o-x0_o+1), 0)
        h_o  = np.maximum((y1_o-y0_o+1), 0)        
        inter_area = w_o * h_o
        
        union_area = a_cs[idx[:,0]] + a_cs[idx[:,1]] - inter_area
        score      = inter_area / union_area
        # picked     = np.where(score > nms_threshold)[0] # these boxes overlap with other boxes
        picked     = (score > nms_threshold)              # these boxes overlap with other boxes
        merge_list = idx[picked,:]                        # list of the overlapping boxes

        # aggregate detections into objects
        merged = np.full((num_rects), False)  # keep track of merged boxes
        for i in range(num_rects):
            if merged[i] == True: continue    # already processed skip
            # objects that need merging
            merge_indx = (merge_list[:,0]==i) # find boxes that overlap with current index i
            merge=merge_list[merge_indx,1]    # these boxes need to be merged
            merged[merge] = True

            if (len(merge) > 1) and use_weighted:
                # merge detections into single object
                total = p_cs[merge].sum()
                rate  = p_cs[merge] / total
                if with_keypoints3D:
                    k = np.array( [
                        [np.sum(kx_cs[merge,:].T * rate)],
                        [np.sum(ky_cs[merge,:].T * rate)],
                        [np.sum(kz_cs[merge,:].T * rate)]
                        ], dtype=np.float32).T
                elif with_keypoints2D:
                    k = np.array( [
                        [np.sum(kx_cs[merge,:].T * rate)],
                        [np.sum(ky_cs[merge,:].T * rate)]
                        ], dtype=np.float32).T
                elif with_keypoints:
                    k = np.array( [
                        [np.sum(kx_cs[merge,:].T * rate)]
                        ], dtype=np.float32).T
                else:
                    k =[]
                obj = Object(\
                    bb = np.array( [
                    [[np.sum(x0_cs[merge] * rate), np.sum(y0_cs[merge] * rate)]],
                    [[np.sum(x1_cs[merge] * rate), np.sum(y1_cs[merge] * rate)]]                         
                    ], dtype=np.float32),
                    l = category,
                    p = total/len(merge),
                    k = k,
                    v = (np.sum( v_cs[merge,:], axis=0) > 0)       if with_visibility else [],
                    type = type)
            else:
                # take highest probability detection
                if with_keypoints3D:
                    k = np.array( [
                        [kx_cs[i,:]],
                        [ky_cs[i,:]],
                        [kz_cs[i,:]]
                        ], dtype=np.float32).T
                elif with_keypoints2D:
                    k = np.array( [
                        [kx_cs[i,:]],
                        [ky_cs[i,:]]
                        ], dtype=np.float32).T
                elif with_keypoints:
                    k = np.array( [
                        [kx_cs[i,:]]
                        ], dtype=np.float32).T
                else:
                    k =[]
                obj = Object(
                    bb = np.array( [
                    [[x0_cs[i],y0_cs[i]]],
                    [[x1_cs[i],y1_cs[i]]]                         
                    ], dtype=np.float32),
                    l = category,
                    p = p_cs[i],
                    k = k,
                    v = v_cs[i,:]  if with_visibility else [],
                    type = type)                
            objects.append(obj)                
    return objects

def nms_weighted(nms_threshold, x0, y0, x1, y1, l, p, kx=[], ky=[], kz=[], v=[], type=objectTypes['rect'], use_weighted=True):
    '''
    The alternative NMS method as mentioned in the BlazeFace paper:
    "We replace the suppression algorithm with a blending strategy that
    estimates the regression parameters of a bounding box as a weighted
    mean between the overlapping predictions."
    Returns a list of objects, one for each detected object.
    '''

    if len(p) == 0: return []

    with_keypoints   = True if (len(kx)>0) else False
    with_keypoints2D = True if (len(ky)>0) else False
    with_keypoints3D = True if (len(kz)>0) else False
    with_visibility  = True if (len(v) >0) else False

    objects = []
    categories = np.unique(l)

    for category in categories:
        # select cordinates of the current category boxes
        picked = (l == category)
        x0_c = x0[picked]
        x1_c = x1[picked]
        y0_c = y0[picked]
        y1_c = y1[picked]
        p_c  =  p[picked]

        if with_keypoints:    kx_c = kx[picked,:]
        if with_keypoints2D:  ky_c = ky[picked,:]
        if with_keypoints3D:  kz_c = kz[picked,:]
        if with_visibility:   v_c  =  v[picked,:] 
                
        remaining_c = np.argsort(p_c)[::-1]

        # merge boxes if we have more than one remaining
        while len(remaining_c) > 0:
            # first box
            first_box=remaining_c[0]
            x0_first = x0_c[first_box]
            y0_first = y0_c[first_box]
            x1_first = x1_c[first_box]
            y1_first = y1_c[first_box]
            p_first  =  p_c[first_box]

            if with_keypoints:    kx_first = kx_c[first_box,:]
            if with_keypoints2D:  ky_first = ky_c[first_box,:]
            if with_keypoints3D:  kz_first = kz_c[first_box,:]
            if with_visibility:   v_first  =  v_c[first_box,:] 
            
            # other boxes (includes first box)
            x0_others = x0_c[remaining_c]
            y0_others = y0_c[remaining_c]
            x1_others = x1_c[remaining_c]
            y1_others = y1_c[remaining_c]
            
            # Intersecton over Union (IoU)

            # areas of the first and other boxes
            area_first   = (x1_first - x0_first + 1) * (y1_first  - y0_first + 1)
            areas_others = (x1_others- x0_others+ 1) * (y1_others - y0_others+ 1)
            # Overlapping (intersection) areas coordinates
            x0_i = np.maximum(x0_first, x0_others)
            y0_i = np.maximum(y0_first, y0_others)
            x1_i = np.minimum(x1_first, x1_others)
            y1_i = np.minimum(y1_first, y1_others)
            # clip at 0 (when there is no overlap)
            w_i  = np.clip((x1_i-x0_i+1), a_min=0, a_max=None)
            h_i  = np.clip((y1_i-y0_i+1), a_min=0, a_max=None)        
            # overlapping area
            area_i = w_i * h_i
            union  = area_first + areas_others - area_i
            # intersection over union
            # small number minimal overlap
            ious  = area_i / union

            # If two detections don't overlap enough, they are considered to be from different objects.
            mask = (ious >= nms_threshold) 
            overlapping_c = remaining_c[mask]   # overlapping
            remaining_c   = remaining_c[~mask]  # not overlap

            # Take an average of the coordinates from the overlapping detections, weighted by their confidence scores.
            if (len(overlapping_c) > 1) and use_weighted:
                total = p_c[overlapping_c].sum()                    
                rate  = p_c[overlapping_c] / total
                if with_keypoints3D:
                    k = np.array( [
                        [np.sum(kx_c[overlapping_c,:].T * rate, axis=1)],
                        [np.sum(ky_c[overlapping_c,:].T * rate, axis=1)],
                        [np.sum(kz_c[overlapping_c,:].T * rate, axis=1)]
                        ], dtype=np.float32).T
                elif with_keypoints2D:
                    k = np.array( [
                        [np.sum(kx_c[overlapping_c,:].T * rate, axis=1)],
                        [np.sum(ky_c[overlapping_c,:].T * rate, axis=1)]
                        ], dtype=np.float32).T
                elif with_keypoints:
                    k = np.array( [
                        [np.sum(kx_c[overlapping_c,:].T * rate, axis=1)]
                        ], dtype=np.float32).T
                else:
                    k =[]
                obj = Object(\
                    bb = np.array( [
                        [[np.sum(x0_c[overlapping_c] * rate), np.sum(y0_c[overlapping_c] * rate)]],
                        [[np.sum(x1_c[overlapping_c] * rate), np.sum(y1_c[overlapping_c] * rate)]]                         
                    ], dtype=np.float32),                   
                    l  = category, 
                    p  = total / len(rate),
                    k  = k,
                    v  = (np.sum( v_c[overlapping_c,:], axis=0) > 0) if with_visibility else [],
                    type = type)
            else:
                if with_keypoints3D:
                    k = np.array( [
                        [kx_first],
                        [ky_first]
                        [kz_first]
                        ], dtype=np.float32).T
                elif with_keypoints2D:
                    k = np.array( [
                        [kx_first],
                        [ky_first]
                        ], dtype=np.float32).T
                elif with_keypoints:
                    k = np.array( [
                        [kx_first]
                        ], dtype=np.float32).T
                else:
                    k =[]
                obj = Object(\
                    bb = np.array( [
                        [[x0_first,y0_first]],
                        [[x1_first,y1_first]]                         
                    ], dtype=np.float32),                   
                    l  = category, 
                    p  = p_first,
                    k  = k,
                    v  = v_first  if with_visibility else [],
                    type = type)
            objects.append(obj)                        
    return objects

def nms_blaze(detections, nms_threshold=0.3):
    '''
    Original code from balze face. 
    This is same as nms_weighted but not accelerated and cleaned up.
    It is left here for reference.
    
    Input: detections are an np array with columns in the format:
        box:       topleft_x, topleft_y, bottomright_x, bottomright_y
        landmarks: x0, y0, x1, y1, x2, y2, x3, y3, x4, y4
        score:     confidence score

    The alternative NMS method as mentioned in the BlazeFace paper:
    "We replace the suppression algorithm with a blending strategy that
    estimates the regression parameters of a bounding box as a weighted
    mean between the overlapping predictions."
    The original MediaPipe code assigns the score of the most confident
    detection to the weighted detection, but we take the average score
    of the overlapping detections.
    The input detections should be a matrix of shape (count, 13).
    Returns a list of objects, one for each detected object.

    This is based on the source code from:
    mediapipe/calculators/util/non_max_suppression_calculator.cc
    mediapipe/calculators/util/non_max_suppression_calculator.proto
    '''

    if len(detections) == 0: return []

    output_detections = np.array([])

    # Sort the detections from highest to lowest score.
    remaining = np.argsort(detections[:, -1])[::-1]

    while len(remaining) > 0:
        detection = detections[remaining[0]]

        # Compute the overlap between the first box and the other remaining boxes. 
        # (Note that the other_boxes also include the first_box.)
        first_box    = detection[:4]
        other_boxes  = detections[remaining, :4]

        # Intersecton over Union (IoU)
        ious = _overlap_similarity(first_box, other_boxes)

        # If two detections don't overlap enough, they are considered to be from different objects.
        mask = (ious >= nms_threshold)
        overlapping = remaining[mask]
        remaining   = remaining[~mask]

        # Take an average of the coordinates from the overlapping detections, weighted by their confidence scores.
        weighted_detection = detection.copy()
        if len(overlapping) > 1:
            coordinates = detections[overlapping, :-1]
            scores      = detections[overlapping, -1:]
            total_score = scores.sum()
            weighted = (coordinates * scores).sum(axis=0) / total_score
            weighted_detection[:-1] = weighted
            weighted_detection[-1]  = total_score / len(overlapping)

        output_detections = np.concatenate((output_detections, weighted_detection))
        
    return output_detections

def _overlap_similarity(box, other_boxes):
    '''
    Computes the IOU between a bounding box and set of other boxes.
    '''
    return _jaccard(np.expand_dims(box,axis=0), other_boxes).squeeze(0)
    # return self.jaccard(box.unsqueeze(0), other_boxes).squeeze(0)

def _jaccard(boxes_a, boxes_b):
    '''
    Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Ground truth bounding boxes,      Shape: [num_objects,4]
        box_b: Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap:                         Shape: [boxes_a.size(0), boxes_b.size(0)]
    '''
    inter = _intersect(boxes_a, boxes_b)
    areas_a = np.broadcast_to((boxes_a[:, 2]-boxes_a[:, 0]) *
                np.expand_dims((boxes_a[:, 3]-boxes_a[:, 1]), axis=1), inter.shape)  # [A,B]
    areas_b = np.broadcast_to((boxes_b[:, 2]-boxes_b[:, 0]) *
                np.expand_dims((boxes_b[:, 3]-boxes_b[:, 1]), axis=0), inter.shape)  # [A,B]
    union = areas_a + areas_b - inter
    return inter / union  # [A,B]

def _intersect(boxes_a, boxes_b):
    '''
    We resize both matrices to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between boxes_a and boxes_b.
    Args:
        boxes_a: bounding boxes, Shape: [A,4].
        boxes_b: bounding boxes, Shape: [B,4].
    Return:
        intersection area, Shape: [A,B].
    '''
    A = boxes_a.shape[0]
    B = boxes_b.shape[0]
    
    max_xy = np.minimum(np.broadcast_to(np.expand_dims(boxes_a[:, 2:], axis=1), (A, B, 2)),
                        np.broadcast_to(np.expand_dims(boxes_b[:, 2:], axis=0), (A, B, 2)))
    min_xy = np.maximum(np.broadcast_to(np.expand_dims(boxes_a[:, :2], axis=1), (A, B, 2)),
                        np.broadcast_to(np.expand_dims(boxes_b[:, :2], axis=0), (A, B, 2)))

    inter = np.clip((max_xy - min_xy), 0, np.inf)
    return inter[:, :, 0] * inter[:, :, 1]

###########################################################
# Feature Classification
###########################################################

def matchEmbeddings(embedding, embeddings, metric):
    """
    Computes distance of representation to all members of embedings database.
    Returns best match and distance to best match.
    Input: embedding of interest, all embeddings to compare to, distance metric
    Output (min_index, min_distance, all distances)
    """
    
    min_distance = 1000 # very large distance
    min_index = 0
    num_embeddings = len(embeddings)
    distances = np.empty(num_embeddings, dtype=np.double)
    for indx in range(num_embeddings): 
        embedding_ref = embeddings[indx]
        if   metric == 'cosine':       d = CosineDistance(                 embedding,               embedding_ref)
        elif metric == 'euclidean':    d = EuclideanDistance(              embedding,               embedding_ref)
        elif metric == 'euclidean_l2': d = EuclideanDistance( l2_normalize(embedding), l2_normalize(embedding_ref))
        distances[indx] = d
        if d < min_distance:
            min_index = indx
            min_distance = d
    return min_index, min_distance, distances

def Zscore(data):
    '''
    Computes Z-score of every element in the data (array)
    (data - mean(data)) / std(data)
    '''
    # origin: UU
    mean, std = cv2.meanStdDev(data)    
    return cv2.divide(cv2.subtract(data,mean.T), std.T)

def CosineDistance(source_representation, test_representation):
    ''' 
    Computes Cosine Distance between two data arrays
    '''
    # origin: serengil
    if type(source_representation) == list: 
        s = np.array(source_representation)
    else: 
        s = source_representation
    if type(test_representation)   == list:   
        t = np.array(test_representation)
    else: 
        t = test_representation
    a = np.matmul(np.transpose(s), t)
    b = np.sum(np.multiply(s, s))
    c = np.sum(np.multiply(t, t))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def EuclideanDistance(source_representation, test_representation):
    ''' 
    Computes Eucledian Distance between two data arrays
    '''
    # origin: serengil
    if type(source_representation) == list: 
        s = np.array(source_representation)
    else:
        s = source_representation
    if type(test_representation) == list:   
        t = np.array(test_representation)
    else:
        t = test_representation
    d = s - t
    euclidean_distance = np.sqrt(np.sum(np.multiply(d, d)))
    return euclidean_distance

def l2_normalize(x):
    # origin: serengil
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def findThreshold(model_name, distance_metric):
    
    base_threshold = {'cosine': 0.40, 'euclidean': 0.55, 'euclidean_l2': 0.75}
    
    thresholds     = {
        'VGG-Face':   {'cosine': 0.40,  'euclidean': 0.60,  'euclidean_l2': 0.86},
        'Facenet':    {'cosine': 0.40,  'euclidean': 10,    'euclidean_l2': 0.80},
        'Facenet512': {'cosine': 0.30,  'euclidean': 23.56, 'euclidean_l2': 1.04},
        'ArcFace':    {'cosine': 0.68,  'euclidean': 4.15,  'euclidean_l2': 1.13},
        'Dlib': 	  {'cosine': 0.07,  'euclidean': 0.6,   'euclidean_l2': 0.4},
        'OpenFace':   {'cosine': 0.10,  'euclidean': 0.55,  'euclidean_l2': 0.55},
        'DeepFace':   {'cosine': 0.23,  'euclidean': 64,    'euclidean_l2': 0.64},
        'DeepID': 	  {'cosine': 0.015, 'euclidean': 45,    'euclidean_l2': 0.17}
    }

    threshold = thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)

    return threshold