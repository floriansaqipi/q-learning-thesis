import numpy as np
import torch
import random

from collections import deque

from .compute_device import ComputeDevice

class ReplayMemory:
    def __init__(self, size: int, batch_size: int):
        self.device = ComputeDevice.get_device()
        self.size = size
        self.batch_size = batch_size
        self.memory = deque(maxlen=size)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self):
        random_transitions_sample = random.sample(self.memory, self.batch_size)
        obs_sample, action_sample, reward_sample, terminated_sample, next_obs_sample = zip(*random_transitions_sample)

        obs_sample = torch.tensor(np.array(obs_sample), dtype=torch.float32, device=self.device)
        reward_sample = torch.tensor(np.array(reward_sample), dtype=torch.float32, device=self.device).unsqueeze(1)
        action_sample = torch.tensor(np.array(action_sample), dtype=torch.long, device=self.device).unsqueeze(1)
        terminated_sample = torch.tensor(np.array(terminated_sample), dtype=torch.bool, device=self.device).unsqueeze(1)
        next_obs_sample = torch.tensor(np.array(next_obs_sample), dtype=torch.float32, device=self.device)

        return obs_sample, action_sample, reward_sample, terminated_sample, next_obs_sample