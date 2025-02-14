import numpy as np
import torch
import torch.nn.functional as functional
import torch.optim as optim

from .agent import Agent
from ..environment import Environment
from .utils import ComputeDevice, FramePreprocessor
from .network import A3CNetwork


class A3CAgent(Agent):
    def __init__(
            self,
            env: Environment,
            learning_rate: float
    ):
        super().__init__(env)
        self.learning_rate = learning_rate
        self.device = ComputeDevice.get_device()
        self.action_space = self.env.get_action_space().n
        self.network = A3CNetwork(self.action_space).to(device=self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.frame_preprocessor = FramePreprocessor()


    def get_action(self, obs):
        obs = self.frame_preprocessor.preprocess(obs)
        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        self.network.eval()
        with torch.no_grad():
            action_values, _ = self.network(obs)
        self.network.train()

        policy = functional.softmax(action_values, dim=-1)
        action_indices = torch.multinomial(policy, 1).squeeze()
        return action_indices.cpu().numpy()


    def get_best_action(self, obs):
        obs = self.frame_preprocessor.preprocess(obs)
        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        self.network.eval()
        with torch.no_grad():
            action_values, _ = self.network(obs)
        self.network.train()

        best_action_indices =  torch.argmax(action_values, dim=1).squeeze()

        return  best_action_indices.cpu().numpy()



def update(self, obs, action, reward, terminated, next_obs):
        pass

    def decay_epsilon(self):
        pass

    def load_progress(self, file_name: str):
        pass

    def save_progress(self, file_name: str):
        pass