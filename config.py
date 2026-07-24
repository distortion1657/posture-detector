# MediaPipe Pose Detection 
MODEL_PATH = './pose_landmarker_heavy.task'
MIN_DETECTION_CONFIDENCE = 0.7
MIN_PRESENCE_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Posture Sensitivity Thresholds
# Leaning: Shoulder length grows by 120% 
leaning_threshold = 1.2

# Slouching: dy drops beyond 80% of baseline
slouching_threshold = 0.8

