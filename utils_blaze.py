###########################################################
# Blaze Utils
# Urs Utzinger, Spring 2023
###########################################################
import numpy as np
import cython

np.seterr(over='ignore')
from scipy.special import expit # is same as sigmoid
def sigmoid_np(x):
    # if len(x) < 400: # is faster for small arrays
    # return(expit(x))
    # else: return(1. / (1. + np.exp(-x)))
    return (1. / (1. + np.exp(-x)))

def decode_boxes(score_thresh, input_img_w, input_img_h, raw_scores, raw_boxes, anchors, anchors_fixed=True):
    '''
    Converts the predictions into actual coordinates using the anchor boxes.
    Keep only the boxes with a confidence score above a threshold.
    out: 
        rect:      topleft_x, topleft_y, bottomright_x, bottomright_y
        landmarks: x0, y0, x1, y1, x2, y2, x3, y3, x4, y4
        score:     confidence score 
    '''
    # scores is cls
    # raw_boxes is reg
    
    sscores = sigmoid_np(raw_scores.flatten()) # cls
    picked = (sscores > score_thresh)      
    scores = sscores[picked]

    # Bounding Box
    # a raw boxe items consists of 12 elements
    # cx, cy, w, h, kx_0, ky_0, kx_1, ky_1, kx_2, ky_2, kx_3, ky_3
    if anchors_fixed:
        # Code if fixed anchor size is used        
        cx = raw_boxes[picked,0] / input_img_w + anchors[picked, 0]
        cy = raw_boxes[picked,1] / input_img_h + anchors[picked, 1]
        w  = raw_boxes[picked,2] / input_img_w
        h  = raw_boxes[picked,3] / input_img_h
    else:
        # Code if variabe anchor size is used
        cx = raw_boxes[picked,0] / input_img_w * anchors[picked, 2] + anchors[picked, 0]
        cy = raw_boxes[picked,1] / input_img_h * anchors[picked, 3] + anchors[picked, 1]
        w  = raw_boxes[picked,2] / input_img_w * anchors[picked, 2]
        h  = raw_boxes[picked,3] / input_img_h * anchors[picked, 3]

    x0 = cx - (w / 2.)  # top left x, xmin
    y0 = cy - (h / 2.)  # top left y, ymin
    x1 = cx + (w / 2.)  # bottom right x, xmax
    y1 = cy + (h / 2.)  # bottom right y, ymax

    # Landmarks original code
    # lx = p[4 + (2 * j) + 0];
    # ly = p[4 + (2 * j) + 1];
    # lx = lx + anchor.x_center*input_img_w
    # ly = ly + anchor.y_center*input_img_h
    # lx = lx/input_img_w
    # ly = ly/input_img_h
    
    if anchors_fixed:
        # Code if fixed anchor size is used
        keypoint_x = raw_boxes[picked,4::2] / input_img_w  + np.expand_dims(anchors[picked, 0], axis=1)
        keypoint_y = raw_boxes[picked,5::2] / input_img_h  + np.expand_dims(anchors[picked, 1], axis=1)
    else:
        # Code if variabe anchor size is used
        keypoint_x = raw_boxes[picked,4::2] / input_img_w * np.expand_dims(anchors[picked, 2], axis=1) + np.expand_dims(anchors[picked, 0], axis=1)
        keypoint_y = raw_boxes[picked,5::2] / input_img_h * np.expand_dims(anchors[picked, 3], axis=1) + np.expand_dims(anchors[picked, 1], axis=1)

    return (x0, y0, x1, y1, keypoint_x, keypoint_y, scores)

###########################################################
# Anchors (Blaze)
###########################################################

class Anchor(object):
    def __init__(self, x_center, y_center, w, h):
        self.x_center = x_center
        self.y_center = y_center
        self.w = w
        self.h = h

class AnchorsParams(object):
    def __init__(self, \
                 num_layers = 5, 
                 input_size_width=224, 
                 input_size_height=224, 
                 min_scale=0.1484375,  
                 max_scale=0.75, 
                 anchor_offset_x=0.5,  
                 anchor_offset_y=0.5, 
                 strides=[8, 16, 32, 32, 32], 
                 aspect_ratios=[1.0], 
                 fixed_anchor_size=True, 
                 interpolated_scale_aspect_ratio=1.0, 
                 reduce_boxes_in_lowest_layer=False):
        
        self.input_size_width                = input_size_width
        self.input_size_height               = input_size_height
        self.min_scale                       = min_scale
        self.max_scale                       = max_scale
        self.anchor_offset_x                 = anchor_offset_x
        self.anchor_offset_y                 = anchor_offset_y
        self.num_layers                      = num_layers
        self.strides                         = strides
        self.aspect_ratios                   = aspect_ratios
        self.fixed_anchor_size               = fixed_anchor_size
        self.interpolated_scale_aspect_ratio = interpolated_scale_aspect_ratio
        self.reduce_boxes_in_lowest_layer    = reduce_boxes_in_lowest_layer

# Likely remove this and use generate_anchors_np          

def generate_anchors(options):
    """
    https://github.com/vidursatija/BlazePalm/blob/master/ML/genarchors.py
    """
    
    layer_id = 0
    anchors = []

    for layer_id, stride in enumerate(options.strides):
        
        anchor_height = []
        anchor_width  = []
        aspect_ratios = []
        scales        = [] 
        
        last_same_stride_layer = layer_id
        while (last_same_stride_layer < len(options.strides)) and \
            (options.strides[last_same_stride_layer] == options.strides[layer_id]):

            scale = _calculate_scale(options.min_scale, options.max_scale,last_same_stride_layer, len(options.strides))
            for aspect_ratio in options.aspect_ratios:
                aspect_ratios.append(aspect_ratio)
                scales.append(scale)
            
            if last_same_stride_layer == len(options.strides) - 1:
                scale_next = 1.0
            else:
                scale_next = _calculate_scale(options.min_scale, options.max_scale,last_same_stride_layer + 1,len(options.strides))
                
            scales.append(sqrt(scale * scale_next))
            aspect_ratios.append(1.0)
            
            last_same_stride_layer += 1

        for (aspect_ratio,scale) in zip(aspect_ratios,scales): 
            ratio_sqrts = sqrt(aspect_ratio)
            anchor_height.append(scale / ratio_sqrts)
            anchor_width.append(scale * ratio_sqrts)

        feature_map_height = 0
        feature_map_width  = 0
        # stride             = options.strides[layer_id]
        feature_map_height = ceil(1.0 * options.input_size_height / stride)
        feature_map_width  = ceil(1.0 * options.input_size_width / stride)

        for y in range(feature_map_height): 
            for x in range(feature_map_width): 
                for a in anchor_height:
                    x_center = (x + options.anchor_offset_x) * 1. / feature_map_width
                    y_center = (y + options.anchor_offset_y) * 1. / feature_map_height

                    new_anchor = Anchor(\
                        x_center=x_center, 
                        y_center=y_center, 
                        w=1.0, 
                        h=1.0)

                    anchors.append(new_anchor)

        layer_id = last_same_stride_layer

    return anchors

# Keep this and call it generate_anchors

def generate_anchors_np(options):
    '''
    https://github.com/vidursatija/BlazePalm/blob/master/ML/genarchors.py
    '''
    strides_size   = len(options.strides)
    assert options.num_layers == strides_size, "num_layers should be equal to the size of strides."

    layer_id = 0
    anchors  = np.array([], dtype=np.float32).reshape(0,4) # empty array 4 elements wide
    
    # For each layer, we create different anchors.
    while layer_id < strides_size:
        anchor_height = []
        anchor_width  = []
        aspect_ratios = []
        scales        = [] 
        
        # For same strides, we merge the anchors in the same order.
        last_same_stride_layer = layer_id
        while (last_same_stride_layer < strides_size) and \
            (options.strides[last_same_stride_layer] == options.strides[layer_id]):
            scale = _calculate_scale(options.min_scale,
                                            options.max_scale,
                                            last_same_stride_layer,
                                            strides_size)
            
            if last_same_stride_layer == 0 and options.reduce_boxes_in_lowest_layer:
                # For first layer, it can be specified to use predefined anchors.
                aspect_ratios.append(1.0)
                aspect_ratios.append(2.0)
                aspect_ratios.append(0.5)
                scales.append(0.1)
                scales.append(scale)
                scales.append(scale)                
            else:
                for aspect_ratio in options.aspect_ratios:
                    aspect_ratios.append(aspect_ratio)
                    scales.append(scale)
                if options.interpolated_scale_aspect_ratio > 0.0:
                    if (last_same_stride_layer == strides_size - 1): 
                        scale_next = 1.0
                    else:
                        scale_next = _calculate_scale(options.min_scale,
                                                            options.max_scale,
                                                            last_same_stride_layer + 1,
                                                            strides_size)
                    scales.append(np.sqrt(scale * scale_next))
                    aspect_ratios.append(options.interpolated_scale_aspect_ratio)
            
            last_same_stride_layer += 1
            
        for aspect_ratio, scale in zip(aspect_ratios, scales):
            ratio_sqrts = np.sqrt(aspect_ratio)
            anchor_height.append(scale / ratio_sqrts)
            anchor_width.append(scale * ratio_sqrts)
            
        stride = options.strides[layer_id]
        feature_map_height = int(np.ceil(options.input_size_height / stride))
        feature_map_width  = int(np.ceil(options.input_size_width / stride))

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for (anchor_w, anchor_h) in zip(anchor_width,anchor_height):
                    x_center = (x + options.anchor_offset_x) / feature_map_width
                    y_center = (y + options.anchor_offset_y) / feature_map_height

                    if options.fixed_anchor_size:
                        new_anchor = np.array([[x_center, y_center, 1.0, 1.0]])
                    else:
                        new_anchor = np.array([[x_center, y_center, anchor_w, anchor_h]])
                        
                    anchors = np.concatenate((anchors, new_anchor),axis=0)
                        
        layer_id = last_same_stride_layer
        
    return anchors

def _calculate_scale(min_scale, max_scale, stride_index, num_strides):
    '''
    https://github.com/vidursatija/BlazePalm/blob/master/ML/genarchors.py
    verified
    '''
    if (num_strides == 1):
        return (min_scale + max_scale) * 0.5
    else:
        return min_scale + (max_scale - min_scale) * (stride_index) / (num_strides - 1)    
