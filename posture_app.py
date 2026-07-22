import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math

# Initialize camera capture object (0 for default webcam)
cap = cv2.VideoCapture(0)

# Media Pipe Initialization
model_path = './pose_landmarker_full.task'
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.7)

# Calibration Variables
baseline_width = None
baseline_dy = None # Difference in distance of nose and shoulders
is_calibrated = False

# Slouch Variables
leaning_threshold = 1.2
slouching_threshold = 0.8
is_slouching = False

if not cap.isOpened():
    print("Error: Webcam not found")
    exit()

print("Webcam connected. Press 'Q' to exit")

def get_shoulder_distance(pt1:tuple, pt2:tuple):
    return math.sqrt((pt2[0]-pt1[0])**2+(pt2[1]-pt1[1])**2)

def get_vertical_gap(nose, lsh, rsh):
    """ Calculates the average height of shoulders, then the delta of shoulder height to nose height"""
    shoulder_y_avg = (lsh[1]+rsh[1])/2

    return shoulder_y_avg - nose[1]


with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        # Read a frame from the camera
        # ret is a boolean, True if frame was captured successfully
        # frame is the actual numpy array
        ret, frame = cap.read()

        

        if not ret:
            print("Failed to capture frame")
            break
        rbg_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = rbg_frame)

        
        frame_timestamp_ms = int(time.time()*1000)
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        if (results.pose_landmarks and len(results.pose_landmarks)>0):
            pose = results.pose_landmarks[0]
            h,w,_ = frame.shape

            nose = pose[0]
            left_shoulder = pose[11]
            right_shoulder = pose[12]

            # Nose Point
            cx, cy = int(nose.x*w),int(nose.y*h)
            cv2.circle(frame, (cx,cy), 3, (0,255,0), -1)

            # Shoulder Points
            lsx, lsy = int(left_shoulder.x*w), int(left_shoulder.y*h)
            rsx, rsy = int(right_shoulder.x*w), int(right_shoulder.y*h)

            cv2.circle(frame, (lsx, lsy), 3, (0,255,0), -1)
            cv2.circle(frame, (rsx, rsy), 3, (0,255,0), -1)

            current_shoulder_width = get_shoulder_distance((lsx,lsy), (rsx,rsy))
            current_dy = get_vertical_gap((cx,cy), (lsx,lsy), (rsx,rsy))

            # Key presses
            key_press = cv2.waitKey(1) & 0xFF
            if key_press == ord('q'):
                break
            elif key_press == ord('c'):
                # Calibration
                baseline_width = current_shoulder_width
                baseline_dy= current_dy
                is_calibrated = True
                
            if is_calibrated:
                cv2.putText(frame, "STATUS: CALIBRATED", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            else: 
                cv2.putText(frame, "PLEASE CALIBRATE. PRESS 'C' TO CALIBRATE.", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            cv2.putText(frame, "Slouching: ", (30,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
            cv2.putText(frame, "YES", (180, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1 ) if is_slouching else cv2.putText(frame, "NO", (180,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1)

            # Slouch check
            if(baseline_dy and baseline_width):
                if (current_shoulder_width > baseline_width * leaning_threshold) or (current_dy < baseline_dy * slouching_threshold):               
                    is_slouching = True
                else:
                    is_slouching = False

            
            cv2.imshow("Posture Detection Engine", frame)


cap.release()
cv2.destroyAllWindows()
