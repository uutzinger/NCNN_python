###########################################################################
# Test Live
###########################################################################
# Origin: Urs Utzinger

import cv2
import time
import logging
from camera.utils import genCapture

from retinaface import RetinaFace
from blur import Blur

# Camera Settings
camera_index    = 0
configs = {
    'camera_res'      : (1280, 720 ),   # width & height
    'exposure'        : -2,             # -1,0 = auto, 1...max=frame interval, 
    'autoexposure'    : -1,             # depends on camera: 0.25 or 0.75(auto) or 1(auto), -1, 0
    'fps'             : 30,             # 15, 30, 40, 90, 120, 180
    'fourcc'          : -1,             # n.a.
    'buffersize'      : -1,             # n.a.
    'output_res'      : (-1, -1),       # Output resolution, -1,-1 no change
    'flip'            : 0,              # 0=norotation 
                                        # 1=ccw90deg 
                                        # 2=rotation180 
                                        # 3=cw90 
                                        # 4=horizontal 
                                        # 5=upright diagonal flip 
                                        # 6=vertical 
                                        # 7=uperleft diagonal flip
    'gain'             : 4,              # camera gain
    'settings'         : 0.0,           # open camera settings
    'displayfps'       : 30             # frame rate for display server
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

# Calculators
# -----------------------------------------------------------------------------
# Face CNN
# find faces, low threshold shows performance of algorithm, make it large for real world application
net_retina = RetinaFace(prob_threshold=0.5, nms_threshold=0.3, use_gpu=False)
# Blur Detector
calc_blur = Blur(3./4.)

# Create and start camera interface
# -----------------------------------------------------------------------------
camera = genCapture(configs, camera_index)
logger.log(logging.INFO, "Getting Images")
camera.start()

# Initialize Variables
# -----------------------------------------------------------------------------
last_time = last_update_time = time.perf_counter()
loop_time = 0.
blur_time = 0.
blur_count = 0
 
faces = []
stop = False
blur_max = 0.0
blur_min = 1.0

# Main Loop
while not stop:
    current_time = time.perf_counter()
    
    # Wait for new image
    (frame_time, frame) = camera.capture.get(block=True, timeout=None)
    # Display camera log messages
    while not camera.log.empty():
        (level, msg) = camera.log.get_nowait()
        logger.log(level, "{}".format(msg))

    if (current_time - last_time) > compute_interval : 
        pipeline_tic = time.perf_counter()
        last_time = current_time

        display_img = frame.copy()

        faceobjects, _ = net_retina(frame, scale=True, use_weighted_nms=False)        
        
        for face in faceobjects:
            face.rotateBoundingBox()

            tic = time.perf_counter()
            pHP, pLP  = calc_blur(frame, face, fft=False)
            blur = (pHP/(pHP+pLP))
            toc = time.perf_counter()
            if blur > blur_max: blur_max = blur
            if blur < blur_min: blur_min = blur
            
            face.draw(display_img)
            if blur < 0.04: # Experimentally determined threshold
                face.printText(display_img, "{:.2f},{:.2f},{:.2f}".format(blur,blur_max,blur_min), color=(0,0,255))
            else:
                face.printText(display_img, "{:.2f},{:.2f},{:.2f}".format(blur,blur_max,blur_min), color=(0,255,0))

        cv2.imshow(window_name, display_img)
        cv2.waitKey(1)
        blur_time += toc-tic
        blur_count += 1

        pipeline_toc = time.perf_counter()
        loop_time = 0.9*loop_time + 0.1*(pipeline_toc - pipeline_tic)

    if (current_time - last_update_time) > update_interval:
        last_update_time =  current_time
        logger.log(logging.INFO,     "Blur        {:.2f} ms".format(1000.*blur_time/blur_count))
        logger.log(logging.INFO,     "Pipeline    {:.2f} ms".format(1000.*loop_time))
            
    try:    
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty(window_name, 0) < 0): stop = True
    except: stop = True  

del net_retina
del blur
camera.stop()
cv2.destroyAllWindows()
