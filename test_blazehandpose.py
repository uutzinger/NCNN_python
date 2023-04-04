
###########################################################################
# Main Testing
###########################################################################
# Origin: Urs Utzinger
#
import cv2
import time
import logging
import time
import numpy as np
from blazepalm import Palm
from blazehandpose import HandLandmarkDetect
from camera.utils import genCapture
from utils_image import extractObjectROI

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
    'displayfps'       : 30             # frame rate for display server
}

# Program Flow
# -----------------------------------------------------------------------------
if configs['displayfps'] >= 0.8*configs['fps']: display_interval = 0
else:                                           display_interval = 1.0/configs['displayfps']

# Throttle the CNN computations    
compute_interval= display_interval

# Averaging time for performance caculations
update_interval = 5.0

# Display
# -----------------------------------------------------------------------------
window_name      = 'Hand Skeleton'
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
logger = logging.getLogger("Blaze Hand")

# Calculators
# -----------------------------------------------------------------------------
# Blaze Objects CNN
handnet = Palm(prob_threshold=0.7, nms_threshold=0.3, level='light', use_gpu=False, num_threads=-1)
skeletonnet = HandLandmarkDetect(prob_threshold=0.4, level='light', use_gpu=False, num_threads=-1)
     
# Create and start camera interface
# -----------------------------------------------------------------------------
camera = genCapture(configs, camera_index)
logger.log(logging.INFO, "Getting Images")
camera.start()

# Initialize Variables
# -----------------------------------------------------------------------------
last_time = last_update_time = last_compute_time = time.perf_counter()
loop_time = 0.

faces = []
stop = False

# Execuation timing
cnn_hand_time  = display_time = lm_time = 0.
cnn_hand_count = 0
display_count = 0
cnn_hand_routines_time  = np.array([0., 0., 0., 0., 0. ])
cnn_skeleton_time = np.array([0., 0., 0.])
cnn_skeleton_count = 0

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
    
    # Assess the image
    if (current_time - last_compute_time) > compute_interval : 
        last_compute_time = current_time
        tic = time.perf_counter()
        hands, times_hand = handnet(frame, scale=True)
        toc = time.perf_counter(); 
        cnn_hand_time  += toc-tic
        cnn_hand_count += 1
        cnn_hand_routines_time  += times_hand
    
        tic = time.perf_counter()
        display_img = frame.copy()
        for hand in hands:
            frame_roi, mat_trans, rotatedObject = extractObjectROI(frame, hand, target_size = 224, simple=True)
            hand, times_landmark = skeletonnet(frame_roi)            
            cnn_skeleton_time  += times_landmark
            cnn_skeleton_count += 1
            hand.invtransform(mat_trans)
            hand.draw(display_img)
        cv2.imshow(window_name, display_img)
        toc = time.perf_counter()
        display_time += toc-tic
        display_count += 1

    loop_toc = time.perf_counter()
    loop_time = 0.9*loop_time + 0.1*(loop_toc - current_time)
    
    # Print Peformance
    if (current_time - last_update_time) > update_interval:
        last_update_time =  current_time
        if cnn_hand_count > 0: 
            logger.log(logging.INFO, "Blaze Hand      {:.2f} ms".format(1000.*cnn_hand_time/cnn_hand_count))
            logger.log(logging.INFO, "  Preprocess    {:.2f} ms".format(1000.*cnn_hand_routines_time[0]/cnn_hand_count))
            logger.log(logging.INFO, "  Extract       {:.2f} ms".format(1000.*cnn_hand_routines_time[1]/cnn_hand_count))
            logger.log(logging.INFO, "  Decode        {:.2f} ms".format(1000.*cnn_hand_routines_time[2]/cnn_hand_count))
            logger.log(logging.INFO, "  NMS           {:.3f} ms".format(1000.*cnn_hand_routines_time[3]/cnn_hand_count))
            logger.log(logging.INFO, "  Create Palm   {:.3f} ms".format(1000.*cnn_hand_routines_time[4]/cnn_hand_count))
        if cnn_skeleton_count > 0:             
            logger.log(logging.INFO, "Landmark Hand")
            logger.log(logging.INFO, "  Preprocess    {:.2f} ms".format(1000.*cnn_skeleton_time[0]/cnn_skeleton_count))
            logger.log(logging.INFO, "  Extract       {:.2f} ms".format(1000.*cnn_skeleton_time[1]/cnn_skeleton_count))
            logger.log(logging.INFO, "  Skeleton      {:.3f} ms".format(1000.*cnn_skeleton_time[2]/cnn_skeleton_count))
        if display_count > 0: 
            logger.log(logging.INFO, "Display         {:.2f} ms".format(1000.*display_time/display_count))
            
        logger.log(logging.INFO,     "Avg Loop        {:.2f} ms".format(1000.*loop_time))
                
        cnn_hand_time  = display_time = lm_time = 0.
        cnn_hand_count = 0
        display_count = 0
        cnn_hand_routines_time  = np.array([0., 0., 0., 0., 0. ])
        cnn_skeleton_time = np.array([0., 0., 0.])
        cnn_skeleton_count = 0

    try:    
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty(window_name, 0) < 0): stop = True
    except: stop = True  
    
del handnet, skeletonnet
camera.stop()
cv2.destroyAllWindows()
