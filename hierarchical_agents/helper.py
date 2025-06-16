import math
from typing import Tuple
import torch


def is_intersection(linea: Tuple[Tuple[float, float]], lineb: Tuple[Tuple[float, float]], get_point=False):

    # get line rep y = a + mx

    m_a = slope_of_line(linea[0], linea[1])
    a_a = linea[0][1] - m_a * linea[0][0]

    m_b = slope_of_line(lineb[0], lineb[1])
    a_b = lineb[0][1] - m_b * lineb[0][0]

    # check for vertical lines
    if m_a == math.inf:
        x_inter = linea[0][0]
        y_inter = a_b + m_b * x_inter
    elif m_b == math.inf:
        x_inter = lineb[0][0]
        y_inter = a_a + m_a * x_inter
    elif m_a == m_b:
        return False
    else:
        x_inter = (a_a-a_b)/(m_b-m_a)
        y_inter = a_a + m_a * x_inter

    # check if intersection point is on line segment
    check = (
                round(min(linea[0][0], linea[1][0]), 2) <= round(x_inter, 2) <= round(max(linea[0][0], linea[1][0]), 2) 
                and round(min(lineb[0][0], lineb[1][0]), 2) <= round(x_inter, 2) <= round(max(lineb[0][0], lineb[1][0]), 2)
                and round(min(linea[0][1], linea[1][1]), 2) <= round(y_inter, 2) <= round(max(linea[0][1], linea[1][1]), 2)
                and round(min(lineb[0][1], lineb[1][1]), 2) <= round(y_inter, 2) <= round(max(lineb[0][1], lineb[1][1]), 2)
            )
    if not check:
        return False
    else:
        # should point be returned
        if get_point:
            return (x_inter, y_inter)
        else:
            # print(f"there is an intersection between: {linea} (wall) and {lineb} (whisker). Intersection point {(x_inter, y_inter)}")
            return True

def relative_delta(angle_degrees: float, radius: float):
    """
    Returns the coordinates (x, y) on circle with radius for a given angle in degrees.
    Gives us the relative position. Center at rob_x, rob_y
    """
    # Ensure the angle is within the range 0 to 360
    angle_degrees = angle_degrees % 360

    # Convert degrees to radians
    angle_radians = math.radians(angle_degrees)

    dx = radius * math.cos(angle_radians)
    dy = radius * math.sin(angle_radians)

    return dx, dy

def relative_angle(center_point, satellite_point):
    """
    returns the angle between a center point and a satellite point. helper function to redirect robot to target
    """

    dx, dy = (satellite_point[0] - center_point[0]) , (satellite_point[1] - center_point[1])

    return math.degrees(math.atan2(dy, dx)) % 360

def distance_between_point(pointa: Tuple[float, float], pointb: Tuple[float, float]):
    if pointa is None or pointb is None or pointa is False or pointb is False:
        return math.inf
    return math.sqrt(math.pow(pointa[0]-pointb[0],2) + math.pow(pointa[1]-pointb[1],2))

def slope_of_line(pointa: Tuple[float, float], pointb: Tuple[float, float]):
    try:
        return (pointb[1] - pointa[1]) / (pointb[0] - pointa[0])
    except ZeroDivisionError:
        return math.inf
    

class RunningStatsState:
    def __init__(self, size):
        self.size = size
        self.mean = torch.zeros(size=(size,))
        self.var = torch.zeros(size=(size,))
        self.count = 0

    def update(self, state):
        self.count += 1
        delta_mean = torch.Tensor(state) - self.mean
        self.mean += delta_mean / self.count
        self.var += (delta_mean*(torch.Tensor(state)-self.mean))/self.count

    def get_mean_std(self):
        std = torch.sqrt(self.var) + 0.001 if self.count > 1 else torch.ones(size=(self.size,))
        return self.mean, std
    
    def standardize(self, state):
        mean, std = self.get_mean_std()
        return (torch.Tensor(state) - mean)/ std
    
    def destandardize(self, state):
        mean, std = self.get_mean_std()
        return (torch.Tensor(state)*std) + mean
