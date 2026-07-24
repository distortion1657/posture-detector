from config import MODEL_PATH, MIN_DETECTION_CONFIDENCE, MIN_PRESENCE_CONFIDENCE, MIN_TRACKING_CONFIDENCE
import mediapipe as mp
import cv2 as cv

class PostureDetector:
    def __init__(self):
        model_path = MODEL_PATH
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode 

        options = PoseLandmarkerOptions(
            base_options = BaseOptions(model_asset_path=model_path),
            running_mode = VisionRunningMode.VIDEO,
            min_pose_detection_confidence = MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence = MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence = MIN_TRACKING_CONFIDENCE,
        )

        self.landmarker = PoseLandmarker.create_from_options(options)

    def extract_landmarks(self, frame, timestamp_ms):
        """Processes a single BGR frame and returns pixel coordinates for (nose, left_shoulder, right_shoulder).
        
        Returns None if no pose is detected.
        """
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, rgb_frame)
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.pose_landmarks is None or len(results.pose_landmarks)<=0:
            return None

        landmarks = results.pose_landmarks[0]
        h,w,_ = frame.shape

        def convert_to_coordinates(x):
            return (int(x.x*w), int(x.y*h))

        nose = convert_to_coordinates(landmarks[0])
        l_sh = convert_to_coordinates(landmarks[11])
        r_sh = convert_to_coordinates(landmarks[12])

        self.draw_landmarks(frame, nose, l_sh, r_sh)
        return nose, l_sh, r_sh

    def draw_landmarks(self, frame, nose, l_sh, r_sh):
        cv.circle(frame, (nose), 3, (0,255,0), -1 )
        cv.circle(frame, (l_sh), 3, (0,255,0), -1 )
        cv.circle(frame, (r_sh), 3, (0,255,0), -1 )

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.landmarker:
            self.landmarker.close()