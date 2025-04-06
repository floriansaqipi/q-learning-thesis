import os

import torch

from .compute_device import ComputeDevice
from .shared_rms_prop import SharedRMSprop
from ..network import A3CNetwork
from ...constans import A3CConstants
from ...environment import Environment
import torch.optim as optim
from .experience_handler import ExperienceHandler

class A3CGlobalNetwork(ExperienceHandler):
    def __init__(self, env: Environment, learning_rate: float, decay_factor: float, epsilon: float):
        self.env = env
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.epsilon = epsilon
        self.device = ComputeDevice.get_device()
        self.network = A3CNetwork(self.env.get_action_space().n).to(device=self.device)
        self.network.share_memory()
        self.optimizer = SharedRMSprop(self.network.parameters(), lr=self.learning_rate, alpha=self.decay_factor, eps=self.epsilon)

    def load_progress(self):
        file_full_path = A3CConstants.PROGRESS_MEMORY_DIRECTORY + A3CConstants.PROGRESS_MEMORY_FILE_NAME
        if not os.path.exists(file_full_path):
            return
        checkpoint = torch.load(file_full_path, weights_only=True)
        self.network.load_state_dict(checkpoint[A3CConstants.CHECKPOINT_NETWORK_STATE_DICT])
        self.optimizer.load_state_dict(checkpoint[A3CConstants.CHECKPOINT_OPTIMIZER_STATE_DICT])

    def save_progress(self, save_frequency: int, episode_number: int = 0, return_queue=None, length_queue=None):
        checkpoint = {
            A3CConstants.CHECKPOINT_NETWORK_STATE_DICT: self.network.state_dict(),
            A3CConstants.CHECKPOINT_OPTIMIZER_STATE_DICT: self.optimizer.state_dict(),
        }
        file_full_path = A3CConstants.PROGRESS_MEMORY_DIRECTORY + A3CConstants.PROGRESS_MEMORY_FILE_NAME
        torch.save(checkpoint, file_full_path)
