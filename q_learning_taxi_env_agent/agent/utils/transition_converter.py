import numpy as np
import torch

from .compute_device import ComputeDevice


class TransitionConverter:
    def __init__(self):
        self.device = ComputeDevice.get_device()

    def to_tensor(self, obs_batch, actions_batch, rewards_batch, terminated_batch, next_obs_batch):

        obs_batch = torch.tensor(np.array(obs_batch), dtype=torch.float32, device=self.device)
        actions_batch = torch.tensor(np.array(actions_batch), dtype=torch.long, device=self.device)
        rewards_batch = torch.tensor(np.array(rewards_batch), dtype=torch.float32, device=self.device)
        terminated_batch = torch.tensor(np.array(terminated_batch), dtype=torch.bool, device=self.device)
        next_obs_batch = torch.tensor(np.array(next_obs_batch), dtype=torch.float32, device=self.device)

        return obs_batch, actions_batch, rewards_batch, terminated_batch, next_obs_batch

    def to_tensor_zip(self, transitions):
        obs_batch, actions_batch, rewards_batch, terminated_batch, next_obs_batch = zip(*transitions)

        obs_batch, actions_batch, rewards_batch, terminated_batch, next_obs_batch = (
            self.to_tensor(
                obs_batch, actions_batch, rewards_batch, terminated_batch, next_obs_batch))

        return obs_batch, actions_batch.unsqueeze(1), rewards_batch.unsqueeze(1), terminated_batch.unsqueeze(1), next_obs_batch

