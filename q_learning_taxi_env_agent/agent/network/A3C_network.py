
import torch
import torch.nn as nn
import torch.nn.functional as functional

from ...hyper_parameters.A2C_hyper_parameters import SEED

class A3CNetwork(nn.Module):
    def __init__(self, action_space_size: int, seed: int = SEED):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.action_space_size = action_space_size

        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.fc1 = nn.Linear(2592, 256)
        self.fc2a = nn.Linear(256, self.action_space_size)
        self.fc2s = nn.Linear(256, 1)


    def forward(self, obs):
        x = functional.relu(self.conv1(obs))
        x = functional.relu(self.conv2(x))

        x = torch.flatten(x, start_dim=1)
        x = functional.relu(self.fc1(x))
        action_values = self.fc2a(x)
        state_value = self.fc2s(x).squeeze(-1)
        return action_values, state_value
