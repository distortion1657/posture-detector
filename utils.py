"""
Point to point geometric math helper functions for landmark analysis.
"""

import math
Point2D = tuple[int, int]

def get_shoulder_distance(l_sh: Point2D,  r_sh: Point2D):
    """
    Calculates the 2D Euclidean distance between two pixel coordinates.

    Args:
        l_sh: Coordinates of the left shoulder
        r_sh: Coordinates of the right shoulder
    
    Returns:
        The Euclidean distance between those two points as float.
    """
    return math.sqrt((r_sh[0]-l_sh[0])**2+(r_sh[1]-l_sh[1])**2)

def get_vertical_gap(nose: Point2D, l_sh: Point2D, r_sh: Point2D):
    """ 
    Calculates the average height of shoulders, then the delta of shoulder height to nose height
    
    Args:
        nose: The coordinates of the nose
        l_sh: The coordinates of the left shoulder
        r_sh: The coordinates of the right shoulder
    
    Returns: 
        A float that is the average vertical gap between the nose and shoulders.
    """
    shoulder_y_avg = (l_sh[1]+r_sh[1])/2
    
    return shoulder_y_avg - nose[1]