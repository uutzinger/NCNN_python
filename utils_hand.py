###########################################################
# Hand Gesture Detection
# Urs Utzinger, Spring 2023
###########################################################
import cv2
import numpy as np
from math import pi

##############################################################
# Hand Keypoints
# 0: wrist
# 1: thumb 1
# 2: thumb 2
# 3: thumb 3
# 4: thumb 4
# 5: index 1
# 6: index 2
# 7: index 3
# 8: index 4
# 9: middle 1
# 10: middle 2
# 11: middle 3
# 12: middle 4
# 13: ring 1
# 14: ring 2
# 15: ring 3
# 16: ring 4
# 17: pinky 1
# 18: pinky 2
# 19: pinky 3
# 20: pinky 4
##############################################################

##############################################################
# Gesture
##############################################################   
# index finger tip loc = hand.k[8]

# Need some adjustments for swaer, three, scissors
#                           point and one
#                           two and victory
# Could incorporate 3D

def gesture(hand):
    
    """ 
    Handpose
        thumb  = 0
        point  = 1
        middle = 2
        ring   = 3
        little = 4

    Direction up = 0, right=1, down=2, left=3
    Curl      open=0, half=1, closed=2
    
    https://github.com/syauqy/handsign-tensorflow 
    https://github.com/andypotato/fingerpose
    """  

    finger_direction, finger_curl = _calcPose(hand)
    # direction_txt = "{: 3.0f},{: 3.0f},{: 3.0f},{: 3.0f},{: 3.0f}".format(finger_direction[0],finger_direction[1],finger_direction[2],finger_direction[3],finger_direction[4])
    # curl_txt = "{: 3.0f},{: 3.0f},{: 3.0f},{: 3.0f},{: 3.0f}".format(finger_curl[0],finger_curl[1],finger_curl[2],finger_curl[3],finger_curl[4])
    
    # C = 0 is opened 0..60
    # C = 1 is half curled 60..120
    # C = 2 is closed 120..180
    C = (finger_curl > 60.)*1 + (finger_curl > 120.)*1 
    # adjust thumb because its harder to curl thumb than the other fingers
    C[0] = (finger_curl[0] > 45.)*1 + (finger_curl[0] > 60.)*1

    # D = 0 is up 65..115 ! smaller range than 45..135
    # D = 1 is rigth -45..65
    # D = 2 is down  -45..-135
    # D = 3 is left 115..180 -135..-180
    D = np.logical_and(finger_direction <  65., finger_direction >=  -45.)*1 +\
        np.logical_and(finger_direction < -45., finger_direction >= -135.)*2 +\
        np.logical_or( finger_direction <-135., finger_direction >=  115.)*3
                
    # Point:
    #  index opened
    #  middle half or closed
    #  ring half or closed
    #  little half or closed
    # D = [-1, 0, -1, -1, -1]
    # C = [>0, 0, >0, >0, >0]
    if C[0] > 0 and C[1] == 0 and C[2] > 0 and C[3] > 0 and C[4] > 0:
        return "point"
    
    # Swaer:
    #  thumb opened
    #  index opened and up
    #  middle opened and up
    #  ring half or closed
    #  little half or closed
    #  index and middle split
    if C[0] == 0 and C[1] == 0 and C[2] == 0 and C[3] > 0 and C[4] > 0 and\
       D[1] == 0 and D[2] == 0 and \
       abs(finger_direction[2] - finger_direction[1]) >= 5.:
       return "swear"
    
    # Thumbs up
    #  thumb opened and up
    #  point closed or half
    #  middle closed or half
    #  ring closed or half
    #  little closed or half
    if C[0] == 0 and C[1] > 0 and C[2] > 0 and C[3] > 0 and C[4] > 0 and \
       D[0] == 0:
       # return  "thumbs up:{} {}".format(direction_txt, curl_txt)
       return  "thumbs up"

    # Thumbs down
    #  thumb opened and down
    #  point closed or half
    #  middle closed or half
    #  ring closed or half
    #  little closed or half
    if C[0] == 0 and C[1] > 0 and C[2] > 0 and C[3] > 0 and C[4] > 0 and \
       D[0] == 2:
       # return "thumbs down:{} {}".format(direction_txt, curl_txt)
       return "thumbs down"

    # Thumbs left
    #  thumb opened and left
    #  point closed or half
    #  middle cloed or half
    #  ring closed or half
    #  little closed or half
    if C[0] == 0 and C[1] > 0 and C[2] > 0 and C[3] > 0 and C[4] > 0 and \
       D[0] == 3:
       # return "thumbs left:{} {}".format(direction_txt, curl_txt)
       return "thumbs left"

    # Thumbs right
    #  thumb right and opened
    #  point closed or half
    #  middle closed or half
    #  ring closed or half
    #  little closed or half
    if C[0] == 0 and C[1] > 0 and C[2] > 0 and C[3] > 0 and C[4] > 0 and \
       D[0] == 1:
       # return "thumbs right:{} {}".format(direction_txt, curl_txt)
       return "thumbs right"

    # Vulcan, needs to be before checking for paper
    #  thumb not care
    #  point opened
    #  middle opened
    #  ring opend
    #  little opened
    #  angle between middle ring > 10
    if C[1] == 0 and C[2] == 0 and C[3] == 0 and C[4] == 0 and \
       abs(finger_direction[3] - finger_direction[2]) > 10. and \
       abs(finger_direction[2] - finger_direction[1]) < 10. and \
       abs(finger_direction[4] - finger_direction[3]) < 10.:
       return "vulcan"

    # Oath
    #  thumb opened up 
    #  point opened up
    #  middle opened up
    #  ring opened up
    #  little opened up
    if C[0] == 0 and C[1] == 0 and C[2] == 0 and C[3] == 0 and C[4] == 0 and \
       D[0] == 0 and D[1] == 0 and D[2] == 0 and D[3] == 0 and D[4] == 0:
       return "oath"

    # Paper
    #  thumb opened
    #  point opened
    #  middle opened
    #  ring opened
    #  little opened
    if C[0] == 0 and C[1] == 0 and C[2] == 0 and C[3] == 0 and C[4] == 0:
       return "paper"
    
    # Scissor
    #  thumb any
    #  point open and left or right
    #  middle open and left or right
    #  ring half or closed
    #  little half or closed
    if C[1] == 0 and C[2] == 0 and C[3] > 0 and C[4] > 0 and \
      (D[1] == 1 or D[2] == 1 or D[1] == 3 or D[2] == 3) and \
       abs(finger_direction[2] - finger_direction[1]) > 10.:
       return "scissor"
    
    # Rock:
    #  thumb half or closed
    #  point half or closed
    #  middle half or closed
    #  ring half or closed
    #  little half or closed
    if C[0] > 0 and C[1] > 0 and C[2] > 0 and C[3] > 0 and C[4] > 0:
       return "rock"
    
    # Victory:
    #  thumb closed or half
    #  point opened and up
    #  middle opened and up
    #  ring closed or half
    #  little close of half
    #  angle point to middle > 10.
    if C[1] == 0 and C[2] == 0 and C[3] > 0 and C[4] > 0 and \
       D[1] == 0 and D[2] == 0 and \
       abs(finger_direction[2] - finger_direction[1]) > 10.:
       return "victory"

    # Finger:
    #  thumb closed or half
    #  point closed or half
    #  middle opened and up
    #  ring closed or half
    #  little close of half
    if C[0] > 0 and C[1] > 0 and C[2] == 0 and C[3] > 0 and C[4] > 0 and \
       D[2] == 0:
       return "finger"

    # Hook:
    #  thumb not care
    #  point opened
    #  middle closed or half
    #  ring closed or half
    #  little opened
    if C[1] == 0 and C[2] > 0 and C[3] > 0 and C[4] == 0:
       return "hook"

    # Pinky:
    #  thumb not care
    #  point closer or half
    #  middle closed or half
    #  ring closed or half
    #  little opened
    if C[1] > 0 and C[2] > 0 and C[3] > 0 and C[4] == 0:
       return "pinky"

    # One, US style: index finger up
    #  thumb closed or half
    #  point opend and up
    #  middle closed or half
    #  ring closed or half
    #  little closed or half
    if C[0] > 0 and C[1] == 0 and C[2] > 0 and C[3] > 0 and C[4] > 0 and\
       D[1] == 0:
       return "one"
    # Is same as point

    # Two US style, index and middle finger up
    #  thumb closed or hald
    #  point opend and up
    #  middle opend and up
    #  ring closed or half
    #  little clsoed or half
    if C[0] > 0 and C[1] == 0 and C[2] == 0 and C[3] > 0 and C[4] > 0 and\
       D[1]== 0 and D[2] == 0:
       return "two"

    # Three
    #  thumb closed
    #  point opend
    #  middle opened and up
    #  ring opened
    #  little closed
    if C[0] > 0 and C[1] == 0 and C[2] == 0 and C[3] == 0 and C[4] > 0 and\
       D[2] == 0:
       return "three"
    # some use swear for 3
    
    # Four
    #  thumb half ore closed
    #  point opend
    #  middle opened
    #  ring opened
    #  little opened
    if C[0] > 0 and C[1] == 0 and C[2] == 0 and C[3] == 0 and C[4] == 0 and\
       D[2] == 0:
       return "four"

    # OK
    #  thumb and point touching
    #  thumb half or closed
    #  point half or closed
    #  middle open 
    #  ring open
    #  little open
    loc_t_t1 = hand.k[4] # thumb tip
    loc_t_t2 = hand.k[3] # thumb joint
    loc_p_t1 = hand.k[8] # point tip
    thumb2point       = cv2.norm(loc_t_t1 - loc_p_t1) # distance between thumb tip and point tip
    reference_length  = cv2.norm(loc_t_t2 - loc_t_t1) # length of thumb segement
    if C[0] > 0 and C[1] > 0 and C[2] == 0 and C[3] == 0 and C[4] == 0 and \
       (thumb2point < 1.5*reference_length):
       return "ok"

    return "none"

def _angle(k_start,k_end):
    ''' 
    Angle between two keypoints (OpenCV)
    atan2 is measuring from horizontal and counter clockwise is positive
    '''
    # 2D only, disregrad depth information
    # for 3D case would need altitude function and make sure depth is scaled and computed correctly

    v = k_end - k_start
    return np.arctan2(-v[:,0,1],v[:,0,0])*180./pi # -dy,dx neg dy because openxc y is top to bottom
    
def _calcPose(hand):
    
    stride = 4
    j = np.arange(5)
    # wrist to first 
    angle_0 = _angle(hand.k[         0,:,:], hand.k[j*stride+1,:,:])
    # first to second
    angle_1 = _angle(hand.k[j*stride+1,:,:], hand.k[j*stride+2,:,:])
    # second to third
    angle_2 = _angle(hand.k[j*stride+2,:,:], hand.k[j*stride+3,:,:])
    # third to 4th
    angle_3 = _angle(hand.k[j*stride+3,:,:], hand.k[j*stride+4,:,:])
    
    # direction is average angle last two finger segments  
    a0 = (angle_3 + angle_2) / 2.
    finger_direction = a0 - (360. * np.floor((a0 + 180.)/(360.)))

    # angle third finger segement minus first finger segment
    a1 = angle_3 - angle_1
    a1 -= (360. * np.floor((a1 + 180.)/(360.)))

    # angle third finger segement minus wrist segment
    a2 = angle_3 - angle_0
    a2 -= (360. * np.floor((a2 + 180.)/(360.)))
        
    finger_curl = np.maximum(np.abs(a1),np.abs(a2))
        
    return finger_direction, finger_curl
