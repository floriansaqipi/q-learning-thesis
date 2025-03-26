import json
import os

import numpy as np
import torch
import torch.nn.functional as functional
import torch.optim as optim

from .agent import Agent
from .utils.json_progress_handler import JsonProgressHandler
from ..environment import Environment,EnvironmentWrapper, StatisticsRecordingEnvironment
from .utils import ComputeDevice, FramePreprocessor, TransitionConverter, ExperienceHandler
from .network import A3CNetwork
from ..constans import  A2CConstants


class A2CAgent(Agent):
    def __init__(
            self,
            env: Environment,
            learning_rate: float,
            discount_factor: float,
            entropy_regularization_coefficient: float,
            single_env: Environment = None
    ):
        super().__init__(env)
        self.single_env = single_env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.entropy_regularization_coefficient = entropy_regularization_coefficient
        self.device = ComputeDevice.get_device()
        self.action_space = self.env.get_action_space().n
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
        obs = self.frame_preprocessor.preprocess_frame(obs)
        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        self.network.eval()
        with torch.no_grad():
            action_values, _ = self.network(obs)
        self.network.train()

        best_action_indices = torch.argmax(action_values, dim=1).squeeze()

        return  best_action_indices.cpu().numpy()

    def update(self, obs, action, reward, terminated, next_obs, truncated=None):
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
        self.training_error.append(total_loss.item())
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def save_progress(self, save_frequency: int = 0, episode_number: int = 0, return_queue=None, length_queue=None):
        checkpoint = {
            A2CConstants.CHECKPOINT_NETWORK_STATE_DICT: self.network.state_dict(),
            A2CConstants.CHECKPOINT_OPTIMIZER_STATE_DICT: self.optimizer.state_dict(),
        }
        file_full_path = A2CConstants.PROGRESS_MEMORY_DIRECTORY + A2CConstants.PROGRESS_MEMORY_FILE_NAME
        torch.save(checkpoint, file_full_path)

        JsonProgressHandler.save_episode_number(save_frequency, episode_number, A2CConstants.PROGRESS_EPISODE_NUMBER_FILE_NAME)

        temp_env = self.single_env
        while isinstance(temp_env, EnvironmentWrapper):
            if isinstance(temp_env, StatisticsRecordingEnvironment):
                JsonProgressHandler.save_statistics(
                    return_queue,
                    length_queue,
                    self.training_error,
                    A2CConstants.PROGRESS_STATISTICS_FILE_NAME
                )
                break
            temp_env = temp_env.unwrap()


    def load_progress(self):
        file_full_path = A2CConstants.PROGRESS_MEMORY_DIRECTORY + A2CConstants.PROGRESS_MEMORY_FILE_NAME
        if not os.path.exists(file_full_path):
            return
        checkpoint = torch.load(file_full_path, weights_only=True)
        self.network.load_state_dict(checkpoint[A2CConstants.CHECKPOINT_NETWORK_STATE_DICT])
        self.optimizer.load_state_dict(checkpoint[A2CConstants.CHECKPOINT_OPTIMIZER_STATE_DICT])
