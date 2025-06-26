import numpy as np

from robot import Rob_body

X_MIN = 0
X_MAX = 100
Y_MIN = 0
Y_MAX = 100
BORDER_WALLS = {((X_MIN, Y_MIN), (X_MAX, Y_MIN)), ((X_MAX, Y_MIN), (X_MAX, Y_MAX)), ((X_MIN, Y_MAX), (X_MAX, Y_MAX)), ((X_MIN, Y_MIN), (X_MIN, Y_MAX))}

def generate_wall(pointa, pointb):
    return (pointa, pointb)

def generate_point(x, y):
    return (x,y)

def generate_random_point(x_min = 1, x_max=99, y_min=1, y_max=99):
    return (np.random.randint(x_min, x_max), np.random.randint(y_min, y_max))

def generate_random_walls(n, x_min = 0, x_max=100, y_min=0, y_max=100):
    """
    generates a set of walls of the form ((start_x, start_y), (end_x, end_y))
    border walls are always integrated
    """

    return BORDER_WALLS.union( {((np.random.randint(x_min, x_max), np.random.randint(y_min, y_max)), (np.random.randint(x_min, x_max), np.random.randint(y_min, y_max))) for _ in range(n)})

def generate_simple_center_wall(length):
    return BORDER_WALLS.union( {(((X_MAX-X_MIN)/2, (Y_MAX-Y_MIN)*(1-length)), ((X_MAX-X_MIN)/2,(Y_MAX-Y_MIN)*length))})

def generate_double_center_wall():

    x_1 = (X_MAX-X_MIN) * np.random.uniform(0.1, 0,4)
    x_2 = (X_MAX-X_MIN) * np.random.uniform(0.6, 0,9)

    l_1 = np.random.uniform(0.5, 0.9)
    l_2 = np.random.uniform(0.5, 0.9)

    return BORDER_WALLS.union( {((x_1, (Y_MAX-Y_MIN)*(1-l_1)), (x_1,(Y_MAX-Y_MIN)*l_1))}, {((x_2, (Y_MAX-Y_MIN)*(1-l_2)), (x_1,(Y_MAX-Y_MIN)*l_2))})

def generate_world(mode=["random", "simple"], n_walls=0, fuel=300):

    if mode == "random":
        walls = generate_random_walls(n_walls)
        goal = generate_random_point()
        starting_point = generate_random_point()
    elif mode == "simple":
        walls = generate_simple_center_wall(0.8)
        rand = np.random.random()
        candidate_1 = generate_point(np.random.randint((X_MAX-X_MIN)*0.66, X_MAX), np.random.randint(Y_MIN, Y_MAX))
        candidate_2 = generate_point(np.random.randint(X_MIN, (X_MAX-X_MIN)*0.33), np.random.randint(Y_MIN, Y_MAX))
        if rand > 0.5:
            goal = candidate_1
            starting_point = candidate_2
        else:
            goal = candidate_2
            starting_point = candidate_1
    elif mode == "double_wall":
        walls = generate_double_center_wall(0.8)
        rand = np.random.random()
        candidate_1 = generate_point(np.random.randint((X_MAX-X_MIN)*0.66, X_MAX), np.random.randint(Y_MIN, Y_MAX))
        candidate_2 = generate_point(np.random.randint(X_MIN, (X_MAX-X_MIN)*0.33), np.random.randint(Y_MIN, Y_MAX))
        if rand > 0.5:
            goal = candidate_1
            starting_point = candidate_2
        else:
            goal = candidate_2
            starting_point = candidate_1
    
    r = Rob_body(w, init_pos=starting_point, fuel_tank=fuel)
    w = Rob_world(walls, goal, r)

    return w, 

class Rob_world():
    def __init__(self, walls = {}, goal=(0,0), robot=None):
        """walls is a set of line segments
                where each line segment is of the form ((x0,y0),(x1,y1))
        """
        self.walls = walls
        self.goal = goal
        self.robot = robot