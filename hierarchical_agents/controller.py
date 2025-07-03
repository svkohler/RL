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

from training_monitor import TrainMonitor
from world import Rob_world, generate_random_walls, generate_point, generate_world
from robot import Rob_body

from helper import RunningStatsState, TimerDecorator, distance_between_point, OptimizedSequenceMemoryBuffer, OptimizedSequenceBuffer, optimzer_wrapper, timer_decorator, update_target_network

from plotting import Plot_env, Plot_metric

from constants import INT_2_DIR, MODEL_DIMENSIONS

GAMMA = 0.95
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999
ACTION_SPACE = [0,1,2,3]
TARGET_ENTROPY = -1

class Rob_controller(): 
    
    def __init__(self, robot, networks, pretrained=True, path_to_weights=None, lr=1e-5, device="cpu"):
       """The lower-level for the middle layer is the body.
       """
       self.device = device
       self.path_to_weights = path_to_weights
       self.robot = robot
       self.close_threshold = 2 # distance that is close enough to arrived

       self.state = None
       self.previous_state = None
       self.current_goal = None
       self.input_size = 6+len(self.robot.whisker_set.set)

       self.networks, self.optimizers = networks

       self.state_stats = RunningStatsState(self.input_size)

       if pretrained:
           self.load_policy_stats(path_to_weights)


    def return_state(self, as_list=False, standardized=False): 
        state = {
            "x_pos": self.robot.rob_x,
            "y_pos": self.robot.rob_y,
            "dir": self.robot.rob_dir,
            "fuel": self.robot.fuel/self.robot.fuel_tank,
            "dx_goal": self.current_goal[0] - self.robot.rob_x,
            "dy_goal": self.current_goal[1] - self.robot.rob_y,
            **{f"whisker_reading_{i}": whisk.dist_2_nearest_wall for i,whisk in enumerate(self.robot.whisker_set.set)}
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

        if self.robot.arrived:
            reward += arr_reward
        
        elif self.robot.crashed:
            reward += crash_pen

        elif self.robot.fuel == 0:
            reward += out_of_fuel_pen

        return reward
    
    # Function to choose action using epsilon-greedy policy
    def select_action(self, ssb, epsilon=0.0, mode="max"):
        # with torch.no_grad():
        # random choice (when you want to skip random choice, then choose epsilon == 0.0 (default))
        if random.random() < epsilon:
            return np.random.choice(ACTION_SPACE)  # Explore
        else:
            state = torch.FloatTensor(ssb[3]).unsqueeze(0).to(self.device)
            
            output = self.networks["actor"](state)

            # choose either the max or according to probabilities
            if mode == "max":
                return torch.argmax(output).item()  # Exploit
            if mode == "probs":
                probabilities = np.array(output.detach().flatten())
                assert np.sum(probabilities) == 1.0, "Sum of probabilites has to be 1!"
                return np.random.choice(ACTION_SPACE, p=probabilities)


    def do(self, world, body, action, sequence_length, standardized):
        """
        the controller goes to a specified goal during inference
        """

        self.current_goal = action["go_to"]

        self.state = self.return_state(as_list=True, standardized=standardized)

        done = False
        state_sequence_buffer = OptimizedSequenceBuffer(sequence_length, self.input_size)

        while not done: 

            state_sequence_buffer.add((None, 0, None, self.state, None))

            action_taken = self.select_action(state_sequence_buffer.content(), epsilon=0)

            # execute that action
            self.robot.do({"steer": INT_2_DIR[action_taken]})

            self.state = self.return_state(as_list=True, standardized=standardized)

            self.check_arrived()

            done = self.robot.fuel == 0 or self.robot.arrived or self.robot.crashed

        self.robot.check_status()

        pl = Plot_env(world, body)

        pl.show_and_close(5)

    def step(self, state_sequence_buffer, epsilon=0.0, action_mode="max"):
        """
        the controller executes a single step with the robot
        returns a tuple (previous_state, action, reward, next_state, done)
        """

        self.previous_state = self.state

        self.state_stats.update(self.return_state(as_list=True))

        action_taken = self.select_action(state_sequence_buffer.content(), epsilon, mode=action_mode)

        self.robot.do({"steer": INT_2_DIR[action_taken]})

        self.state = self.return_state(as_list=True)

        reward = self.reward()

        self.check_arrived()

        done = self.robot.fuel == 0 or self.robot.arrived or self.robot.crashed

        return self.previous_state, action_taken, reward, self.state, done

    @timer_decorator
    def episode(
            self, 
            world="simple", 
            n_walls = 3, 
            fuel=300, 
            sequence_length=1,
            state_sequence_buffer=None, 
            memory_replay_buffer=None,
            loss_and_update_function=None,
            update_target_network_function=None,
            epsilon=1.0,
            action_mode="max",
            ):
        """
        the controller executes a whole episode with the robot during training
        """
        # generate world for episode
        w, self.robot = generate_world(mode=world, n_walls=n_walls, fuel=fuel)
        self.current_goal = w.goal

        # init state
        self.state = self.return_state(as_list=True)

        # empty sequence buffer (no sequences over episode boundaries)
        self.state_sequence_buffer.empty()

        # initialize run vars
        done = False
        episode_reward = 0
        episode_steps = 0

        while not done:
            ps, a, r, ns, d = self.step(state_sequence_buffer, epsilon=epsilon, action_mode=action_mode)
            done = d
            episode_reward += r
            episode_steps += 1

            # fill sequence and memory buffer
            self.state_sequence_buffer.add((ps, a, r, ns, d))
            if len(state_sequence_buffer) == sequence_length:
                memory_replay_buffer.add(self.state_sequence_buffer.content())
            
            rand = random.random()
            if rand < 0.2:
                loss_and_update_function()
                # if rand < 0.0001:
                #     update_target_network_function()
            update_target_network_function()
        
        if rand < 0.004:
            # pl = Plot_env(w, self.robot)
            self.save_policy_stats(self.path_to_weights + "/tmp/")

        return episode_reward, episode_steps
        
    def train_dqn(
            self, 
            batch_size=64, 
            episodes=1280, 
            memory_length=10000, 
            epsilon=1.0, 
            sequence_length=1,
            n_walls=3, 
            world="simple",
            fuel=300
        ):

        def compute_loss_and_update():
            if len(self.memory_replay_buffer) < batch_size:
                return
            previous_states, actions, rewards, next_states, dones = self.memory_replay_buffer.sample(batch_size)

            # Compute Q-values for current states
            q_values = self.networks["actor"](previous_states).squeeze().gather(1, actions[:, -1].unsqueeze(1))

            # Compute target Q-values using the target network
            with torch.no_grad():
                max_next_q_values = self.networks["target"](next_states).squeeze().max(1)[0]
                target_q_values = rewards[:, -1].squeeze() + GAMMA * max_next_q_values * (1 - dones[:, -1].squeeze())

            loss = nn.MSELoss()(q_values.squeeze(), target_q_values)

            optimzer_wrapper(self.optimizers["actor"], loss)

        def update_target_networks():
            self.networks["target"].load_state_dict(self.networks["actor"].state_dict())
            self.epsilon = max(EPSILON_MIN, (self.epsilon * EPSILON_DECAY))
            print(f"new epsilon: {self.epsilon}")
        
        # set the exploitation-exploration parameter
        self.epsilon = epsilon

        # set the policy in training mode
        self.networks["actor"].train()
        self.networks["target"].eval()
        
        self.memory_replay_buffer = OptimizedSequenceMemoryBuffer(memory_length, sequence_length, self.input_size, self.device)
        self.state_sequence_buffer = OptimizedSequenceBuffer(sequence_length, self.input_size)
        train_monitor = TrainMonitor()

        while train_monitor.num_episodes <= episodes:
            e_r, e_s = self.episode(world, 
                                    n_walls, 
                                    fuel, 
                                    sequence_length, 
                                    self.state_sequence_buffer, 
                                    self.memory_replay_buffer, 
                                    compute_loss_and_update, 
                                    update_target_networks,
                                    epsilon=self.epsilon)
            train_monitor.update(e_r, e_s, self.episode.get_execution_time())
            self.robot.check_status(id=train_monitor.num_episodes, reward=e_r)                


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
                self.robot = Rob_body(w, init_pos=generate_point(), fuel_tank=max_length)

                # empty trajectory to collect states, actions, rewards
                trajectory = []

                self.policy.reset()

                while self.robot.fuel > 0 and not self.robot.arrived and not self.robot.crashed:
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
                        self.robot.do({"steer": INT_2_DIR[action_taken]})

                        self.check_arrived()

                        # get new state after action (normalized)
                        self.state = self.return_state(as_list=True)

                        state, reward = (self.state, self.reward())
                        
                        trajectory.append((state, action_taken, reward))

                states, actions, rewards = zip(*trajectory)
                
                self.robot.check_status(id=number_of_sims, reward=np.sum(rewards))

                # fill batch_tensors from the back in order to clip it efficiently
                batch_states[i,(max_length-len(states)):,:] = torch.tensor(states, dtype=torch.float32)
                batch_actions[i,(max_length-len(actions)):] = torch.tensor(actions, dtype=torch.int64)
                batch_rewards[i,(max_length-len(rewards)):] = torch.tensor(rewards, dtype=torch.float32)

                if len(trajectory) < min_length_traj:
                    min_length_traj = len(trajectory)

                if number_of_sims % 250 == 0:      
                    pl = Plot_env(w, self.robot)

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
                    
    def train_actor_critic(
            self, 
            batch_size=64, 
            episodes=1280, 
            memory_length=10000, 
            sequence_length=1,
            n_walls=3, 
            world="simple",
            fuel=300
    ):
        def compute_loss_and_update():
            if len(self.memory_replay_buffer) < batch_size:
                return
            previous_states, actions, rewards, next_states, dones = self.memory_replay_buffer.sample(batch_size)

            # Update critic
            with torch.no_grad():
                next_probabilities = self.networks["actor"](next_states)
                next_action_values1 = self.networks["target1"](next_states)
                next_action_values2 = self.networks["target2"](next_states)
                next_action_values = torch.min(next_action_values1, next_action_values2)
                next_state_values = (next_probabilities * (next_action_values - self.alpha * torch.log(next_probabilities + 1e-6))).sum(dim=-1)
                target_q_values = rewards + GAMMA * (1 - dones) * next_state_values

            current_q1 = self.networks["critic1"](previous_states).squeeze(1).gather(1, actions).squeeze(-1)
            current_q2 = self.networks["critic2"](previous_states).squeeze(1).gather(1, actions).squeeze(-1)
            critic1_loss = nn.MSELoss()(current_q1, target_q_values)
            critic2_loss = nn.MSELoss()(current_q2, target_q_values)

            optimzer_wrapper(self.optimizers["critic1"], critic1_loss)
            optimzer_wrapper(self.optimizers["critic2"], critic2_loss)

            # Update Actor
            probs = self.networks["actor"](previous_states)
            q1 = self.networks["critic1"](previous_states)
            q2 = self.networks["critic2"](previous_states)
            min_q = torch.min(q1, q2)
            actor_loss = (probs * (self.alpha * torch.log(probs + 1e-6) - min_q)).sum(dim=-1).mean()

            optimzer_wrapper(self.optimizers["actor"], actor_loss)

            # Update Alpha
            entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1).mean()
            alpha_loss = -(self.networks["log_alpha"] * (entropy + TARGET_ENTROPY).detach()).mean()

            optimzer_wrapper(self.optimizers["log_alpha"], alpha_loss)

            self.alpha = self.networks["log_alpha"].exp()

        def update_target_networks():
            for target_source in [(self.networks["target1"], self.networks["critic1"]), (self.networks["target2"], self.networks["critic2"])]:
                update_target_network(target_source[0], target_source[1], mode="soft")

        self.alpha = self.networks["log_alpha"].exp()

        # set the policy in training mode
        self.networks["actor"].train()
        self.networks["critic1"].train()
        self.networks["critic2"].train()
        self.networks["target1"].eval()
        self.networks["target2"].eval()

        self.memory_replay_buffer = OptimizedSequenceMemoryBuffer(memory_length, sequence_length, self.input_size, self.device)
        self.state_sequence_buffer = OptimizedSequenceBuffer(sequence_length, self.input_size)
        train_monitor = TrainMonitor()

        while train_monitor.num_episodes <= episodes:
            e_r, e_s = self.episode(world, 
                                    n_walls, 
                                    fuel, 
                                    sequence_length, 
                                    self.state_sequence_buffer, 
                                    self.memory_replay_buffer, 
                                    compute_loss_and_update, 
                                    update_target_networks, 
                                    action_mode="prob"
                                    )
            train_monitor.update(e_r, e_s, self.episode.get_execution_time())
            self.robot.check_status(id=train_monitor.num_episodes, reward=e_r)                
            
    

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
        current_percept = self.robot.return_state()
        location_rob = (current_percept["rob_x_pos"], current_percept["rob_y_pos"])
        if distance_between_point(location_rob, self.current_goal) < self.close_threshold:
            self.robot.arrived = True

    def save_policy_stats(self, path):

        if not os.path.exists(path):
            Path(path).mkdir(parents=True, exist_ok=True)

        for k, v in self.networks.items():
            if not isinstance(v, torch.Tensor):
                torch.save(self.networks[k].state_dict(), path+f"/{k}_policy_weights.pth")

        with open(path+f"/state_stats.pkl", 'wb') as handle:
            pickle.dump(self.state_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"saved weights and stats to {path}")

    def load_policy_stats(self, path):
        with open(path+"/state_stats.pkl", 'rb') as handle:
            self.state_stats = pickle.load(handle)
        for k, v in self.networks.items():
            self.networks[k].load_state_dict(torch.load(path+f"/{k}_policy_weights.pth", weights_only=True))
            self.networks[k].eval()
        print(f"loaded weights and stats from {path}")

