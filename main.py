import cv2 as cv
import PostureTracker 
import PostureDetector
import time
from utils import get_shoulder_distance, get_vertical_gap


tracker = PostureTracker.PostureTracker()
detector = PostureDetector.PostureDetector()
capture = cv.VideoCapture(0)

while True:
    # Read a frame from the camera
    # ret is a boolean, True if frame was captured successfully
    # frame is the actual numpy array
    ret,frame = capture.read()
    
    if not ret:
        print("Error: Failed to capture frame")
        break

    key_press = cv.waitKey(1) & 0xFF
    if key_press == ord('q'):
        break
    timestamp_ms = int(time.time()*1000) 
    
    landmarks = detector.extract_landmarks(frame, timestamp_ms)
    if landmarks is None:
        cv.putText(frame, "Failed to detect landmarks. Please ensure you are facing the camera and are in a relatively well lit room", (30,100), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    else:
        nose, l_sh, r_sh = landmarks

        if key_press == ord('c'):
            tracker.calibrate(nose, l_sh, r_sh)
        
        if not tracker.is_calibrated:
            cv.putText(frame, "PLEASE CALIBRATE. PRESS 'C' TO CALIBRATE.", (30,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        else:
            cv.putText(frame, "STATUS: CALIBRATED", (30,50), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        current_shoulder_width = get_shoulder_distance(l_sh, r_sh)
        current_dy = get_vertical_gap(nose, l_sh, r_sh)

        is_slouching = tracker.evaluate(current_shoulder_width, current_dy)
        cv.putText(frame, "Slouching: ", (30,90), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        cv.putText(frame, "YES", (180, 90), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1 ) if is_slouching else cv.putText(frame, "NO", (180,90), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1)
        
    cv.imshow("Posture Detection Engine", frame)
