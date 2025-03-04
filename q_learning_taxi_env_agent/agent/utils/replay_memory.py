import numpy as np
import torch
import random

from collections import deque

from .compute_device import ComputeDevice
from .transition_converter import TransitionConverter

class ReplayMemory:
    def __init__(self, size: int, batch_size: int):
        self.device = ComputeDevice.get_device()
        self.size = size
        self.batch_size = batch_size
        self.memory = deque(maxlen=size)
        self.transitions_converter = TransitionConverter()

    def append(self, transition):
        self.memory.append(transition)

    def sample(self):
        random_transitions_sample = random.sample(self.memory, self.batch_size)

        obs_sample, action_sample, reward_sample, terminated_sample, next_obs_sample = zip(*random_transitions_sample)

        obs_sample, action_sample, reward_sample, terminated_sample, next_obs_sample = (
            self.transitions_converter.to_tensor(
                obs_sample, action_sample, reward_sample, terminated_sample, next_obs_sample
            ))

        return obs_sample, action_sample.unsqueeze(1), reward_sample.unsqueeze(1), terminated_sample.unsqueeze(1), next_obs_sample