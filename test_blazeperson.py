
###########################################################################
# Test Hand Pose
###########################################################################
# Origin: Urs Utzinger

import cv2
import time
import logging
import time
import numpy as np

from blazeperson import Person

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
# -----------------------------------------------------------------------------
compute_interval= display_interval

# Averaging time for performance caculations
update_interval = 5.0

# Display
# -----------------------------------------------------------------------------
window_name      = 'Blaze Person'
font             = cv2.FONT_HERSHEY_SIMPLEX
textLocation0    = (10,20)
textLocation1    = (10,60)
fontScale        = 1
fontColor        = (255,255,255)
lineType         = 2
opacity          = 0.2
#
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # or WINDOW_NORMAL


# Calculators
# -----------------------------------------------------------------------------
# Person CNN

personnet   = Person(prob_threshold=0.5, nms_threshold=0.3, use_gpu=False)

# Create and start camera interface
# -----------------------------------------------------------------------------
camera = genCapture(configs, camera_index)
logger.log(logging.INFO, "Getting Images")
camera.start()

# Initialize Variables
# -----------------------------------------------------------------------------
last_time = last_update_time = time.perf_counter()
loop_time = 0.

stop = False

# Execuation timing
cnn_time  = display_time = 0.
cnn_count = 0
cnn_routines_time  = np.array([0., 0., 0., 0., 0.])

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
        last_time = current_time

        pipeline_tic = time.perf_counter()

        display_img = frame.copy()

        tic = time.perf_counter()
        persons, portraits, times_person = personnet(frame, scale=True)
        toc = time.perf_counter(); 

        cnn_time += toc-tic
        cnn_count += 1
        cnn_routines_time += times_person
        
        tic = time.perf_counter()
        for person in persons: 
            person.draw(display_img)

        cv2.imshow(window_name, display_img)
        cv2.waitKey(1)
        toc = time.perf_counter()
        display_time += toc-tic

        pipeline_toc = time.perf_counter()
        loop_time = 0.9*loop_time + 0.1*(pipeline_toc - pipeline_tic)
                
    if (current_time - last_update_time) > update_interval:
        last_update_time =  current_time
        if cnn_count > 0: 
            logger.log(logging.INFO, "Find Person       {:.2f} ms".format(1000.*cnn_time/cnn_count))
            logger.log(logging.INFO, "  Preprocess    {:.2f} ms".format(1000.*cnn_routines_time[0]/cnn_count))
            logger.log(logging.INFO, "  Extract       {:.2f} ms".format(1000.*cnn_routines_time[1]/cnn_count))
            logger.log(logging.INFO, "  Decode        {:.2f} ms".format(1000.*cnn_routines_time[2]/cnn_count))
            logger.log(logging.INFO, "  NMS           {:.2f} ms".format(1000.*cnn_routines_time[3]/cnn_count))
            logger.log(logging.INFO, "  Create Person {:.3f} ms".format(1000.*cnn_routines_time[4]/cnn_count))
            logger.log(logging.INFO, "Display         {:.2f} ms".format(1000.*display_time/cnn_count))
        logger.log(logging.INFO,     "Pipeline        {:.2f} ms".format(1000.*loop_time))
        
        cnn_time  = 0.
        cnn_count = 0
        display_time = 0.
        cnn_routines_time  = np.array([0., 0., 0., 0., 0.])

    try:    
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty(window_name, 0) < 0): stop = True
    except: stop = True  
    
del personnet
camera.stop()
cv2.destroyAllWindows()
