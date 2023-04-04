
###########################################################################
# Test Hand Pose
###########################################################################
# Origin: Urs Utzinger

import cv2
import time
import logging
import time
import numpy as np

from utils import drawHandObjects
from handpose import HandPose

from fastestdet import FastestDet

from camera.utils import genCapture

# Camera Settings
# -----------------------------------------------------------------------------
camera_index    = 0
configs = {
    'camera_res'      : (1280, 720 ),   # width & height
    'exposure'        : -2,             # -1,0 = auto, 1...max=frame interval, 
    'autoexposure'    : 1,             # depends on camera: 0.25 or 0.75(auto) or 1(auto), -1, 0
    'wb_temp'         : 4600,           #
    'autowb'          : 1,              #    
    'fps'             : 30,             # 15, 30, 40, 90, 120, 180
    'fourcc'          : -1,             # n.a.
    'buffersize'      : -1,             # n.a.
    'output_res'      : (-1, -1),       # Output resolution, -1,-1 no change
    'flip'            : 6,              # 0=norotation 
                                        # 1=ccw90deg 
                                        # 2=rotation180 
                                        # 3=cw90 
                                        # 4=horizontal 
                                        # 5=upright diagonal flip 
                                        # 6=vertical 
                                        # 7=uperleft diagonal flip
    'gain'             : 4,             # camera gain
    'settings'         : 0.0,           # open camera settings
    'displayfps'       : 10             # frame rate for display server
}

# Program Flow
# -----------------------------------------------------------------------------
if configs['displayfps'] >= 0.8*configs['fps']: display_interval = 0
else:                                           display_interval = 1.0/configs['displayfps']

# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Objects")

# Throttle the CNN computations    
compute_interval= display_interval

# Averaging time for performance caculations
update_interval = 5.0

# Display
# -----------------------------------------------------------------------------
window_name      = 'Objects'
font             = cv2.FONT_HERSHEY_SIMPLEX
textLocation0    = (10,20)
textLocation1    = (10,60)
fontScale        = 1
fontColor        = (255,255,255)
lineType         = 2
pivot_size       = 112 # face recognition successful match image
opacity          = 0.2
#
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL

# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG) # options are: DEBUG, INFO, ERROR, WARNING
logger = logging.getLogger("Objects")

# Calculators
# -----------------------------------------------------------------------------
# Object CNN
net = FastestDet(prob_threshold=0.8, nms_threshold=0.5, use_gpu=True)

# Create and start camera interface
# -----------------------------------------------------------------------------
camera = genCapture(configs, camera_index)
logger.log(logging.INFO, "Getting Images")
camera.start()

# Initialize Variables
# -----------------------------------------------------------------------------
last_time = last_update_time = time.perf_counter()
loop_time = 0.

faces = []
stop = False

# Execuation timing
ff_time  = display_time = ges_time = 0.
ff_count = 0
ss_time  = np.array([0., 0., 0., 0., 0., 0., 0., 0.])

# #############################################################################
# #############################################################################
# Main Loop
# #############################################################################
# #############################################################################
while (not stop):
    current_time = time.perf_counter()

    # Wait for New Image
    # ----------------------------------------------------------------
    (frame_time, frame) = camera.capture.get(block=True, timeout=None)
    # Display camera log messages
    while not camera.log.empty():
        (level, msg) = camera.log.get_nowait()
        logger.log(level, "{}".format(msg))
    
    if (current_time - last_time) > compute_interval : 
        pipeline_tic = time.perf_counter()
        last_time = current_time

        display_img = frame.copy()

        tic = time.perf_counter()
        hands, times = net(frame, scale=True)
        toc = time.perf_counter(); 
        ff_time += toc-tic
        ff_count += 1
        ss_time += times
        
        tic = time.perf_counter()
        drawHandObjects(display_img, hands, (255,255,255))
        for hand in hands: 
            tic_ges = time.perf_counter()
            ges, l = hand.gesture()
            toc_ges = time.perf_counter()
            ges_time += toc_ges-tic_ges
            cv2.circle(display_img, (int(l.x), int(l.y)), 4, (  0, 255, 255), -1)
            logger.log(logging.INFO, "Gesture       {}".format(ges))
        cv2.imshow(window_name, display_img)
        cv2.waitKey(1)
        toc = time.perf_counter()
        display_time += toc-tic

        pipeline_toc = time.perf_counter()
        loop_time = 0.9*loop_time + 0.1*(pipeline_toc - pipeline_tic)
                
    if (current_time - last_update_time) > update_interval:
        last_update_time =  current_time
        if ff_count > 0: 
            logger.log(logging.INFO, "Find Hand       {:.2f} ms".format(1000.*ff_time/ff_count))
            logger.log(logging.INFO, "  Preprocess    {:.2f} ms".format(1000.*ss_time[0]/ff_count))
            logger.log(logging.INFO, "  Proposals     {:.2f} ms".format(1000.*ss_time[1]/ff_count))
            logger.log(logging.INFO, "    Extract     {:.2f} ms".format(1000.*ss_time[6]/ff_count))
            logger.log(logging.INFO, "    Generation  {:.2f} ms".format(1000.*ss_time[7]/ff_count))
            logger.log(logging.INFO, "  Sort          {:.3f} ms".format(1000.*ss_time[2]/ff_count))
            logger.log(logging.INFO, "  Select        {:.3f} ms".format(1000.*ss_time[3]/ff_count))
            logger.log(logging.INFO, "  Resize        {:.3f} ms".format(1000.*ss_time[4]/ff_count))
            logger.log(logging.INFO, "  Landmarks     {:.3f} ms".format(1000.*ss_time[5]/ff_count))
            logger.log(logging.INFO, "Display         {:.2f} ms".format(1000.*display_time/ff_count))
            logger.log(logging.INFO, "  Gesture       {:.2f} ms".format(1000.*ges_time/ff_count))
        logger.log(logging.INFO,     "Pipeline        {:.2f} ms".format(1000.*loop_time))
        
        ff_time  = 0.
        ff_count = 0
        display_time = 0.
        ss_time  = np.array([0., 0., 0., 0., 0., 0., 0., 0.])

    try:    
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty(window_name, 0) < 0): stop = True
    except: stop = True  
    
del net
camera.stop()
cv2.destroyAllWindows()
