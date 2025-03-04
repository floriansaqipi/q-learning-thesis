import numpy as np
import torch
import torch.nn.functional as functional
import torch.optim as optim

from .agent import Agent
from ..environment import Environment
from .utils import ComputeDevice, FramePreprocessor, TransitionConverter
from .network import A3CNetwork
from ..constans import Constants


class A3CAgent(Agent):
    def __init__(
            self,
            env: Environment,
            learning_rate: float,
            discount_factor: float,
            entropy_regularization_coefficient: float
    ):
        super().__init__(env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.entropy_regularization_coefficient = entropy_regularization_coefficient
        self.device = ComputeDevice.get_device()
        self.action_space = self.env.get_action_space()[0].n
        self.network = A3CNetwork(self.action_space).to(device=self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.frame_preprocessor = FramePreprocessor()
        self.transition_converter = TransitionConverter()


    def get_action(self, obs):
        obs = self.frame_preprocessor.preprocess_frames(obs)
        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
        self.network.eval()
        with torch.no_grad():
            action_values, _ = self.network(obs)
        self.network.train()

        policy = functional.softmax(action_values, dim=-1)
        action_indices = torch.multinomial(policy, 1).squeeze()
        return action_indices.cpu().numpy()


    def get_best_action(self, obs):
        obs = self.frame_preprocessor.preprocess_frames(obs)
        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device)
        self.network.eval()
        with torch.no_grad():
            action_values, _ = self.network(obs)
        self.network.train()

        best_action_indices = torch.argmax(action_values, dim=1).squeeze()

        return  best_action_indices.cpu().numpy()

    def update(self, obs, action, reward, terminated, next_obs):
        obs = self.frame_preprocessor.preprocess_frames(obs)
        next_obs = self.frame_preprocessor.preprocess_frames(next_obs)
        batch_size = len(obs)

        obs_batch, action_batch, reward_batch, terminated_batch, next_obs_batch = (
            self.transition_converter.to_tensor(obs, action, reward, terminated, next_obs)
        )

        self.learn(obs_batch, action_batch, reward_batch, terminated_batch, next_obs_batch, batch_size)


    def learn(self, obs_batch, action_batch, reward_batch, terminated_batch, next_obs_batch, batch_size):

        action_values, state_value = self.network(obs_batch)
        _, next_state_value = self.network(next_obs_batch)
        target_state_value = reward_batch + ((~terminated_batch) * self.discount_factor * next_state_value)

        advantage = target_state_value - state_value
        probabilities = functional.softmax(action_values, dim = -1)
        log_probabilities = functional.log_softmax(action_values, dim = -1)
        entropy = -torch.sum(probabilities * log_probabilities, dim = -1)

        batch_indexes = torch.arange(batch_size)
        log_probabilities_actions = log_probabilities[batch_indexes, action_batch]

        actor_loss = -(log_probabilities_actions * advantage.detach()).mean() - (self.entropy_regularization_coefficient * entropy.mean())
        critic_loss = functional.mse_loss(target_state_value.detach(), state_value)
        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def save_progress(self, file_name: str):
        file_full_path = Constants.PROGRESS_MEMORY_DIRECTORY + file_name
        torch.save(self.network.state_dict(), file_full_path)

    def load_progress(self, file_name: str):
        file_full_path = Constants.PROGRESS_MEMORY_DIRECTORY + file_name
        self.network.load_state_dict(torch.load(file_full_path, weights_only=True))