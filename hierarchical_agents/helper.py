import math
from typing import Tuple
import torch
import numpy as np
import random
import time
from functools import wraps

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


class OptimizedSequenceMemoryBuffer:
    def __init__(self, capacity, sequence_length, state_dim, device="cpu"):
        """
        Optimized memory buffer for sequences.

        Args:
        - capacity (int): Maximum number of sequences the buffer can hold.
        - sequence_length (int): Length of each sequence.
        - state_dim (int): Dimensionality of the state.
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.state_dim = state_dim
        self.size = 0
        self.index = 0
        self.device = device

        # Preallocate memory for the buffer
        self.previous_states = np.zeros((capacity, sequence_length, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, sequence_length), dtype=np.int64)
        self.rewards = np.zeros((capacity, sequence_length), dtype=np.float32)
        self.next_states = np.zeros((capacity, sequence_length, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, sequence_length), dtype=np.float32)

    def add(self, sequence):
        """
        Add a sequence of experiences to the buffer.

        Args:
        - sequence (tuple): A tuple containing (previous_states, actions, rewards, next_states, dones).
          Each component must have a shape of (sequence_length, ...).
        """
        previous_states, actions, rewards, next_states, dones = sequence

        # Store the sequence in the preallocated arrays
        self.previous_states[self.index] = previous_states
        self.actions[self.index] = actions
        self.rewards[self.index] = rewards
        self.next_states[self.index] = next_states
        self.dones[self.index] = dones

        # Update the index and size
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Sample a batch of sequences from the buffer.

        Args:
        - batch_size (int): Number of sequences to sample.

        Returns:
        - A tuple of tensors (previous_states, actions, rewards, next_states, dones).
        """
        indices = random.sample(range(self.size), batch_size)

        # Use indexing to quickly fetch the batch
        previous_states = torch.tensor(self.previous_states[indices], dtype=torch.float32).to(self.device)
        actions = torch.tensor(self.actions[indices], dtype=torch.long).to(self.device)
        rewards = torch.tensor(self.rewards[indices], dtype=torch.float32).to(self.device)
        next_states = torch.tensor(self.next_states[indices], dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.dones[indices], dtype=torch.float32).to(self.device)

        return previous_states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Get the current size of the buffer.

        Returns:
        - int: Number of sequences currently stored in the buffer.
        """
        return self.size


class OptimizedSequenceBuffer:
    def __init__(self, capacity, state_dim, device="cpu"):
        """
        Optimized sequence buffer.

        Args:
        - capacity (int): Length of sequences the buffer holds.
        - state_dim (int): Dimensionality of the state.
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.size = 0
        self.index = 0
        self.device = device

        # Preallocate memory for the buffer
        self.previous_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity), dtype=np.int64)
        self.rewards = np.zeros((capacity), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity), dtype=np.float32)

    def add(self, step_tuple):
        """
        Add a step tuple of to the sequence buffer.

        Args:
        - step_tuple (tuple): A tuple containing (previous_state, action, reward, next_state, done).
        """
        previous_states, actions, rewards, next_states, dones = step_tuple

        # Store the sequence in the preallocated arrays
        self.previous_states[self.index] = previous_states
        self.actions[self.index] = actions
        self.rewards[self.index] = rewards
        self.next_states[self.index] = next_states
        self.dones[self.index] = dones

        # Update the index and size
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def content(self):
        """
        Return the content of the buffer.

        Returns:
        - A tuple (previous_states, actions, rewards, next_states, dones).
        """


        return self.previous_states, self.actions, self.rewards, self.next_states, self.dones
    
    def empty(self):
        self.size = 0
        self.index = 0

        # Preallocate memory for the buffer
        self.previous_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity), dtype=np.int64)
        self.rewards = np.zeros((self.capacity), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity), dtype=np.float32)


    def __len__(self):
        """
        Get the current size of the buffer.

        Returns:
        - int: Number of sequences currently stored in the buffer.
        """
        return self.size

class TimerDecorator:
    def __init__(self, func):
        self.func = func
        self.execution_time = None # Store the time taken for the function

    def __call__(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = self.func(*args, **kwargs)
        end_time = time.perf_counter()
        self.execution_time = end_time - start_time
        return result

    def get_execution_time(self):
        return self.execution_time

def timer_decorator(func):
    execution_time = {"time": None}  # Use a mutable object to store execution time

    @wraps(func)  # Preserve original function metadata
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time["time"] = end_time - start_time
        return result

    # Add a method to retrieve the execution time
    def get_execution_time():
        return execution_time["time"]

    # Attach the method to the wrapper function
    wrapper.get_execution_time = get_execution_time

    return wrapper


def optimzer_wrapper(optimizer, loss):
    """
    wrap lines into a function such that they dont have to be repeated multiple times
    """
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def update_target_network(target, source, mode=["hard", "soft"], tau=0.005):
    """
    function updates target with parameters from a source network
    return: None
    """
    if mode=="hard":
        target.load_state_dict(source.state_dict())
    elif mode == "soft":
        target_state_dict = target.state_dict()
        source_state_dict = source.state_dict()
        for key in source_state_dict:
            target_state_dict[key] = source_state_dict[key]*tau + target_state_dict[key]*(1-tau)



            