
import torch
import torch.nn as nn
import torch.nn.functional as functional

from ...hyper_parameters.A3C_hyper_parameters import SEED

class A3CNetwork(nn.Module):
    def __init__(self, action_space_size: int, seed: int = SEED):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.action_space_size = action_space_size

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3a = nn.Linear(256, self.action_space_size)
        self.fc3s = nn.Linear(256, 1)


    def forward(self, obs):
        x = functional.relu(self.bn1(self.conv1(obs)))
        x = functional.relu(self.bn2(self.conv2(x)))
        x = functional.relu(self.bn3(self.conv3(x)))

        x = torch.flatten(x, start_dim=1)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        action_values = self.fc3a(x)
        state_value = self.fc3s(x).squeeze(-1)
        return action_values, state_value
