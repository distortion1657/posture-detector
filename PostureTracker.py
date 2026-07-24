from utils import get_shoulder_distance, get_vertical_gap, Point2D
from config import slouching_threshold, leaning_threshold
class PostureTracker:
    def __init__(self):
        """ Initialize variables """
        self.baseline_shoulder_width = None
        self.baseline_dy = None # Distance from nose to average height of shoulders
        self.is_calibrated = False
        self.is_slouching = False
        self.slouching_threshold = slouching_threshold
        self.leaning_threshold = leaning_threshold

    def calibrate(self, nose: Point2D, l_sh: Point2D, r_sh: Point2D)-> None:
        """ Calibrate base posture. In the future, use machine learning and stuff
        
        Args:
            nose: A tuple containing the coordinates of the nose
            l_sh: A tuple containing the coordinates of the left shoulder
            r_sh: A tuple containing the coordinates of the right shoulder

            """
        self.baseline_shoulder_width = get_shoulder_distance(l_sh, r_sh)
        self.baseline_dy = get_vertical_gap(nose, l_sh, r_sh)
        print(self.baseline_dy, ",", self.baseline_shoulder_width)
        self.is_calibrated = True
    
    def evaluate(self, current_shoulder_width, current_dy)->bool:
        """ Evaluate current posture. 
        If either the current shoulder width is longer than 120% of baseline shoulder width or the difference in the average height of shoulders and nose is less than 80% of its baseline then
        trigger is_slouching as true.

        Args:
            current_shoulder_width: The euclidean distance between two shoulder points.
            current_dy: The delta of current avg. shoulder height and nose.

        Return:
            is_slouching: A boolean to determine if user is slouching or not. 
        """
        if not self.is_calibrated:
            self.is_slouching = False
            return self.is_slouching

        print("Slouching?", current_dy, self.baseline_dy * slouching_threshold)
        print("Leaning?", current_shoulder_width , self.baseline_shoulder_width * leaning_threshold )
        if current_dy < self.baseline_dy * slouching_threshold or current_shoulder_width > self.baseline_shoulder_width * leaning_threshold:
            self.is_slouching = True
        else:
            self.is_slouching = False
        return self.is_slouching




