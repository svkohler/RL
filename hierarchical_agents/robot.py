import numpy as np

from helper import distance_between_point, is_intersection, relative_delta

class Whisker():

    def __init__(self, length, relative_dir, host):
        self.length = length
        self.relative_dir = relative_dir
        self.host = host
        self.dist_2_nearest_wall = None

    def distance_to_nearest_wall(self):
        if self.host.crashed:
            self.dist_2_nearest_wall = 0
        else:
            distances = [
                distance_between_point(
                    is_intersection(wall, 
                                    (
                                        (self.host.rob_x, self.host.rob_y), 
                                        tuple(map(sum, zip((self.host.rob_x, self.host.rob_y), (relative_delta(self.host.rob_dir+self.relative_dir, self.length)))))
                                    ),
                                    get_point=True
                                ),
                    (self.host.rob_x, self.host.rob_y)
                )  
                for wall in self.host.world.walls
            ]

            min = np.min(distances)

            if min < self.length:
                self.dist_2_nearest_wall = min
            else:
                self.dist_2_nearest_wall = self.length
        
class WhiskerSet():

    def __init__(self, angles, lengths, host):
        self.set = [Whisker(len, ang, host) for ang, len in zip(angles, lengths)]

class Rob_body():
    def __init__(self, world, init_pos=(0,0), initial_dir=90, whisker_angles = [-60, -30, 0, 30, 60], whisker_lengths = [5,5,5,5,5], fuel_tank=2000):
        """ world is the current world
        init_pos is a triple of (x-position, y-position, direction)
        direction is in degrees; 0 is to right, 90 is straight-up, etc
        """
        self.world = world

        self.rob_x, self.rob_y = init_pos
        self.rob_dir = initial_dir

        self.turning_angle = 18 # degrees that a left makes
        self.step_size = 2 # distance to in rob_dir per step
        self.fuel_tank = fuel_tank

        self.crashed = False
        self.arrived = False
        self.fuel = fuel_tank

        self.whisker_set = WhiskerSet(whisker_angles, whisker_lengths, self)
        for whisk in self.whisker_set.set:
            whisk.distance_to_nearest_wall()

        self.route = [(self.rob_x, self.rob_y, self.rob_dir)] # history of (x,y) positions

        # The following control how it is plotted
        self.plotting = True # whether the trace is being plotted
        self.sleep_time = 0.05 # time between actions (for real-timetting)

    def restart(self, pos=(0,0), fuel_tank=2000):
        self.rob_x, self.rob_y = pos
        self.lower.crashed = False
        self.lower.arrived = False
        self.lower.fuel = fuel_tank

    def return_state(self, as_list=False):
        state = {
            'rob_x_pos':self.rob_x, 
            'rob_y_pos':self.rob_y,
            'rob_dir':self.rob_dir, 
            'crashed':self.crashed,
            'arrived': self.arrived,
            'fuel': self.fuel/self.fuel_tank,
            }
        if as_list:
            return list(state.values())
        return state
        

    def do(self, action):
        """ action is {'steer':direction}
        direction is 'left', 'right' or 'straight'.
        Returns current percept.
        """

        self.fuel -= 1

        if action["steer"] == "left":
            self.rob_dir = (self.rob_dir + self.turning_angle) % 360
        elif action["steer"] == "right":
            self.rob_dir =  (self.rob_dir - self.turning_angle) % 360
        elif action["steer"] == "back":
            self.rob_dir =  (self.rob_dir - 180) % 360
        else:
            dx, dy = relative_delta(self.rob_dir, self.step_size)
            # check if this steps leads to a crash
            if any(
                [
                    is_intersection(wall, 
                                    ((self.rob_x, self.rob_y), tuple(map(sum, zip((self.rob_x, self.rob_y), (dx, dy))))))
                    for wall in self.world.walls
                ]
            ):
                self.crashed = True
            self.rob_x = self.rob_x + dx
            self.rob_y = self.rob_y + dy
            self.route.append((self.rob_x, self.rob_y, self.rob_dir))

        for whisk in self.whisker_set.set:
            whisk.distance_to_nearest_wall()

    def check_status(self, id=None, action=None, reward=None):

        string = ""

        if id is not None:
            string += f"{id}: "

        if self.fuel == 0:
            string += "FAILURE, the robot ran out of fuel! Destination not reached!"
        elif self.crashed:
            string += f"FAILURE, The robot crashed!!! Fuel remaining: {self.fuel}."
        elif self.arrived:
            string += f"SUCCESS!!! The robot arrived at the destination! Congrats! Fuel remaining: {self.fuel}."

        if reward is not None:
            string += f" Reward collected during this episode: {round(reward, 2)}."

        print(string)