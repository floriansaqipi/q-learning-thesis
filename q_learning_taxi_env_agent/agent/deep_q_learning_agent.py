from typing_extensions import override

import numpy as np
import torch
import torch.nn.functional as functional
import torch.optim as optim

from .agent import Agent
from .network import DeepQLearningNetwork
from .utils import ReplayMemory, ComputeDevice
from ..environment import Environment
from ..constans import Constants

class DeepQLearningAgent(Agent):

    def __init__(
            self,
            env: Environment,
            learning_rate: float,
            discount_factor: float,
            interpolation_parameter: float,
            start_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            replay_memory: ReplayMemory
    ):
        super().__init__(env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.interpolation_parameter = interpolation_parameter
        self.device = ComputeDevice.get_device()
        self.observation_space_size = env.get_observation_space().shape[0]
        self.action_space_size = env.get_action_space().n
        self.q_network = DeepQLearningNetwork(self.observation_space_size, self.action_space_size).to(self.device)
        self.target_q_network = DeepQLearningNetwork(self.observation_space_size, self.action_space_size).to(
            self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_memory = replay_memory

    def get_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(obs)
        self.q_network.train()
        if np.random.random() < self.epsilon:
            return torch.randint(0, action_values.size(1), (1,)).item()
        else:
            return torch.argmax(action_values, dim=1).item()

    def get_best_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(obs)
        self.q_network.train()
        return torch.argmax(action_values, dim=1).item()

    def update(self, obs, action, reward, terminated, next_obs):
        self.replay_memory.append((obs, action, reward, terminated, next_obs))
        if len(self.replay_memory.memory) > self.replay_memory.batch_size:
            transitions = self.replay_memory.sample()
            self.learn(transitions)
            self.soft_update()

    def learn(self, transitions):
        obs_batch, actions_batch, reward_batch, terminated_batch, next_obs_batch = transitions
        future_q_target_values, _ = self.target_q_network(next_obs_batch).detach().max(dim=1, keepdim=True)
        q_target_values = reward_batch + ((~terminated_batch) * self.discount_factor * future_q_target_values)
        q_values = self.q_network(obs_batch).gather(1, actions_batch)
        loss = functional.mse_loss(q_values, q_target_values)
        self.training_error.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self):
        for target_parameter, local_parameter in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_parameter.data.copy_(
                self.interpolation_parameter * local_parameter +
                    (1.0 - self.interpolation_parameter) * target_parameter)

    def save_progress(self, file_name: str):
        file_full_path = Constants.PROGRESS_MEMORY_DIRECTORY + file_name
        torch.save(self.q_network.state_dict(), file_full_path)

    def load_progress(self, file_name: str):
        file_full_path = Constants.PROGRESS_MEMORY_DIRECTORY + file_name
        self.q_network.load_state_dict(torch.load(file_full_path, map_location=self.device, weights_only=True))

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def end_of_episode_hook(self):
        self.decay_epsilon()