import os
import time
from pathlib import Path
import torch
import pickle
from collections import deque
import random
import copy
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

from world import Rob_world, generate_random_walls, generate_point, generate_world
from robot import Rob_body

from helper import RunningStatsState, distance_between_point, OptimizedSequenceMemoryBuffer, OptimizedSequenceBuffer

from plotting import Plot_env, Plot_metric

from constants import INT_2_DIR

class Rob_controller(): 
    
    def __init__(self, lower, policy, pretrained=True, path_to_weights=None, lr=1e-5, device="cpu"):
       """The lower-level for the middle layer is the body.
       """
       self.device = device
       self.path_to_weights = path_to_weights
       self.lower = lower
       self.state = None
       self.previous_state = None
       self.current_goal = None
       self.input_size = 6+len(self.lower.whisker_set.set)

       self.policy = policy.to(device)
       self.state_stats = RunningStatsState(self.input_size)

       if pretrained:
           self.load_policy_stats(path_to_weights)

       self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
       self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "max", 0.5, 1000, cooldown=500, eps=1e-10)

       self.close_threshold = 2 # distance that is close enough to arrived

    def return_state(self, as_list=False, standardized=False): 
        state = {
            "x_pos": self.lower.rob_x,
            "y_pos": self.lower.rob_y,
            "dir": self.lower.rob_dir,
            "fuel": self.lower.fuel/self.lower.fuel_tank,
            "dx_goal": self.current_goal[0] - self.lower.rob_x,
            "dy_goal": self.current_goal[1] - self.lower.rob_y,
            **{f"whisker_reading_{i}": whisk.dist_2_nearest_wall for i,whisk in enumerate(self.lower.whisker_set.set)}
        }
        if as_list:
            if standardized:
                return list(self.state_stats.standardize(list(state.values())))
            else:
                return list(state.values())
        return state
    
    def reward(self, fuel_usage_pen=-.1, crash_pen=-100, out_of_fuel_pen=-100, arr_reward=100, distance_scaling_factor=2):

        # reward is negatively initialized to incentivize not staying in the same spot
        reward = fuel_usage_pen

        old_distance_to_goal = distance_between_point((self.previous_state[0], self.previous_state[1]), self.current_goal)
        new_distance_to_goal = distance_between_point((self.state[0], self.state[1]), self.current_goal)

        # base reward is such that moving closer to target is rewarded
        diff = (old_distance_to_goal - new_distance_to_goal) * distance_scaling_factor
        if diff < 0:
            reward += 2*diff
        else:
            reward += diff

        if self.lower.arrived:
            reward += arr_reward
        
        elif self.lower.crashed:
            reward += crash_pen

        elif self.lower.fuel == 0:
            reward += out_of_fuel_pen

        return reward
    
    # Function to choose action using epsilon-greedy policy
    def select_action(self, state_sequence_buffer, action_space=[0,1,2,3], epsilon=0.0, mode="max"):
        # with torch.no_grad():
        # random choice (when you want to skip random choice, then choose epsilon == 0.0 (default))
        if random.random() < epsilon:
            return np.random.choice(action_space)  # Explore
        else:
            state = torch.FloatTensor(state_sequence_buffer[3]).unsqueeze(0).to(self.device)
            
            output = self.policy(state)

            # choose either the max or according to probabilities
            if mode == "max":
                return torch.argmax(output).item()  # Exploit
            if mode == "probs":
                probabilities = np.array(output.detach().flatten())
                assert np.sum(probabilities) == 1.0, "Sum of probabilites has to be 1!"
                return np.random.choice(action_space, p=probabilities)


    def do(self, world, body, action, sequence_length, standardized):

        self.current_goal = action["go_to"]

        self.state = self.return_state(as_list=True, standardized=standardized)

        done = False
        state_sequence_buffer = deque(maxlen=sequence_length)

        while not done: 

            state_sequence_buffer.append([None, None, None, self.state, None])

            action_taken = self.select_action(state_sequence_buffer, epsilon=0)

            # execute that action
            self.lower.do({"steer": INT_2_DIR[action_taken]})

            self.state = self.return_state(as_list=True, standardized=standardized)

            self.check_arrived()

            done = self.lower.fuel == 0 or self.lower.arrived or self.lower.crashed

        self.lower.check_status()

        pl = Plot_env(world, body)

        pl.show_and_close(5)
        

    def train_dqn(self, 
                batch_size=64, 
                simulations=1280, 
                memory_length=10000, 
                epsilon=1.0, 
                epsilon_min=0.01, 
                epsilon_decay=0.999, 
                sequence_length=1,
                n_walls=3, 
                fuel=300, 
                standardized=False,
                world="sinple"
                ):
        # set the policy in training mode
        self.policy.train()

        # initialize the target network and load weights from policy network
        self.target_network = copy.deepcopy(self.policy).to(self.device)
        self.target_network.load_state_dict(self.policy.state_dict())
        self.target_network.eval()
        
        # init variables to keep track of metrics/simulations
        number_of_sims = 0
        total_steps = 0
        metric_coll = []
        metric_coll_ma = []

        memory_replay_buffer = OptimizedSequenceMemoryBuffer(memory_length, sequence_length, self.input_size, self.device)
        state_sequence_buffer = OptimizedSequenceBuffer(sequence_length, self.input_size)

        while number_of_sims <= simulations:

            number_of_sims += 1

            state_sequence_buffer.empty()

            # each simulation has a specific goal, set of walls and initial starting point
            w, starting_point = generate_world(mode=world, n_walls=n_walls)
            self.current_goal = w.goal

            self.lower = Rob_body(w, init_pos=starting_point, fuel_tank=fuel)

            self.state = self.return_state(as_list=True, standardized=standardized)

            self.policy.reset()

            # initialize run vars
            done = False
            episode_reward = 0
            episode_steps = 0

            while not done:

                total_steps += 1
                episode_steps += 1

                self.previous_state = self.state

                self.state_stats.update(self.return_state(as_list=True))

                action_taken = self.select_action(state_sequence_buffer.content(), epsilon=epsilon)

                # execute that action
                self.lower.do({"steer": INT_2_DIR[action_taken]})

                self.state = self.return_state(as_list=True, standardized=standardized)

                reward = self.reward()

                episode_reward += reward

                self.check_arrived()

                done = self.lower.fuel == 0 or self.lower.arrived or self.lower.crashed

                state_sequence_buffer.add((self.previous_state, action_taken, reward, self.state, done))

                if len(state_sequence_buffer) == sequence_length:
                    memory_replay_buffer.add(state_sequence_buffer.content())

                # compute loss each 5th step
                if total_steps % 2 == 0:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    loss = self.compute_dqn_loss(memory_replay_buffer, batch_size)

                    if loss is not None:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    torch.cuda.synchronize()
                    print(f"time for loss and backprob: {round(time.time()- start_time, 5)}")


                # update target network after 10k steps
                if total_steps % 10000 == 0:
                    self.target_network.load_state_dict(self.policy.state_dict())
                    epsilon = max(epsilon_min, (epsilon * epsilon_decay))


            self.lower.check_status(id=number_of_sims, reward=episode_reward)

            metric_coll.append(episode_reward / episode_steps)

            if number_of_sims > 100:
                start_time = time.time()
                metric_coll_ma.append((sum(metric_coll[-1000:])) / min(len(metric_coll), 1000))
                pmetric = Plot_metric(metric_coll_ma, y_label="step reward", x_label="episodes", title="avg. reward per step")

            if number_of_sims % 250 == 0:      
                start_time = time.time()
                pl = Plot_env(w, self.lower)
                self.save_policy_stats(self.path_to_weights + "tmp/")
                print(f"Current learning rate: {self.lr_scheduler.get_last_lr()}.")
                print(f"Current epsilon: {epsilon}.")




    def train_reinforce(self, batch_size=64, simulations=1280, max_length=2000):

        # set the policy in training mode
        self.policy.train() 

        number_of_sims = 0
        return_per_timestep = []

        while number_of_sims <= simulations:

            # prepare tensors to collect states, actions, rewards over a batch. size limited to max_length (i.e fuel capacity)
            batch_states = torch.zeros(size=(batch_size, max_length, self.input_size), dtype=torch.float32).to(self.device)
            batch_actions = torch.zeros(size=(batch_size, max_length), dtype=torch.int64).to(self.device)
            batch_rewards = torch.zeros(size=(batch_size, max_length), dtype=torch.float32).to(self.device)
            min_length_traj = max_length

            for i in range(batch_size):
                
                number_of_sims += 1

                # each simulation has a specific goal, set of walls and initial starting point
                self.current_goal  = generate_point()
                w = Rob_world(walls = generate_random_walls(0), goal=self.current_goal)
                self.lower = Rob_body(w, init_pos=generate_point(), fuel_tank=max_length)

                # empty trajectory to collect states, actions, rewards
                trajectory = []

                self.policy.reset()

                while self.lower.fuel > 0 and not self.lower.arrived and not self.lower.crashed:
                    with torch.no_grad():

                        # save the old state in order to calculate reward later on
                        self.previous_state = torch.tensor(self.return_state(as_list=True), dtype=torch.float32)

                        # update running stats and normalize input
                        self.state_stats.update(self.previous_state)
                        mean, std = self.state_stats.get_mean_std()
                        previous_state_norm = (self.previous_state-mean)/std

                        # get action probailities and new hidden_state
                        action_probs = self.policy(previous_state_norm.unsqueeze(0))

                        # sample an action from said probabilities
                        action_taken = np.random.choice([0, 1, 2, 3], p=np.array(action_probs.detach().flatten()))

                        # execute that action
                        self.lower.do({"steer": INT_2_DIR[action_taken]})

                        self.check_arrived()

                        # get new state after action (normalized)
                        self.state = self.return_state(as_list=True)

                        state, reward = (self.state, self.reward())
                        
                        trajectory.append((state, action_taken, reward))

                states, actions, rewards = zip(*trajectory)
                
                self.lower.check_status(id=number_of_sims, reward=np.sum(rewards))

                # fill batch_tensors from the back in order to clip it efficiently
                batch_states[i,(max_length-len(states)):,:] = torch.tensor(states, dtype=torch.float32)
                batch_actions[i,(max_length-len(actions)):] = torch.tensor(actions, dtype=torch.int64)
                batch_rewards[i,(max_length-len(rewards)):] = torch.tensor(rewards, dtype=torch.float32)

                if len(trajectory) < min_length_traj:
                    min_length_traj = len(trajectory)

                if number_of_sims % 250 == 0:      
                    pl = Plot_env(w, self.lower)

            return_per_timestep.append(torch.mean(batch_rewards))
            ploss = Plot_metric(return_per_timestep[-2000:])

            self.policy.reset()

            loss = self.compute_loss(
                self.policy, 
                batch_states[:,(max_length-min_length_traj):,:], 
                batch_actions[:,(max_length-min_length_traj):], 
                batch_rewards[:,(max_length-min_length_traj):]
                )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step(return_per_timestep[-1])
            print(f"current leanring rate: {self.lr_scheduler.get_last_lr()}.")
                    
    def compute_dqn_loss(self, memory_replay, batch_size, gamma = 0.95):
        if len(memory_replay) < batch_size:
            return
        previous_states, actions, rewards, next_states, dones = memory_replay.sample(batch_size)

        # Compute Q-values for current states
        q_values = self.policy(previous_states).squeeze().gather(1, actions[:, -1].unsqueeze(1))

        # Compute target Q-values using the target network
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).squeeze().max(1)[0]
            target_q_values = rewards[:, -1].squeeze() + gamma * max_next_q_values * (1 - dones[:, -1].squeeze())

        return nn.MSELoss()(q_values.squeeze(), target_q_values)

    def compute_reinforce_loss(self, policy, states, actions, rewards, gamma=0.99):
        """
            Compute the loss for an LSTM-based policy.

            Args:
                policy: The LSTM policy network (a PyTorch model).
                trajectory: A list of (state, action, reward) tuples.
                gamma: Discount factor.

            Returns:
                loss: The policy loss (scalar).
        """

        discount_factors = torch.Tensor([gamma**i for i in range(rewards.shape[1])]).to(self.device)
        discount_matrix = torch.zeros(size=(len(discount_factors), len(discount_factors))).to(self.device)
        for i in range(len(discount_factors)):
            discount_matrix[i:, i] = discount_factors[:len(discount_factors)-i]

        returns = torch.matmul(rewards, discount_matrix)

        # Forward pass through the LSTM policy network
        mean, std = self.state_stats.get_mean_std()
        action_probs = policy((states-mean)/std)  # Shape: [batch_size, seq_len, action_dim]

        # Compute log probabilities of the actions taken
        action_log_probs = torch.log(torch.gather(action_probs, dim=2, index=actions.unsqueeze(-1))).squeeze(-1)

        # Compute the policy loss
        loss = -torch.sum(action_log_probs * returns)  # Negative of the policy gradient
        return loss

    def check_arrived(self):
        """ middle layer also checks whether roboter has arrived at said destination
        """
        current_percept = self.lower.return_state()
        location_rob = (current_percept["rob_x_pos"], current_percept["rob_y_pos"])
        if distance_between_point(location_rob, self.current_goal) < self.close_threshold:
            self.lower.arrived = True

    def save_policy_stats(self, path):

        if not os.path.exists(path):
            Path(path).mkdir(parents=True, exist_ok=True)

        torch.save(self.policy.state_dict(), path+"/policy_weights.pth")
        with open(path+f"/state_stats.pkl", 'wb') as handle:
            pickle.dump(self.state_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"saved weights and stats to {path}")

    def load_policy_stats(self, path):
        with open(path+"/state_stats.pkl", 'rb') as handle:
            self.state_stats = pickle.load(handle)
        self.policy.load_state_dict(torch.load(path+"/policy_weights.pth", weights_only=True))
        self.policy.eval()
        print(f"loaded weights and stats from {path}")

