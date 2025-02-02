import torch
import torch.nn as nn
import torch.nn.functional as functional

from ...hyper_parameters.convolutional_deep_q_learning_hyper_parameters import SEED


class ConvolutionalDeepQLearningNetwork(nn.Module):
    def __init__(self, action_space_size: int, seed: int = SEED):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.action_space_size = action_space_size

        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(12_800, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.action_space_size)

    def forward(self, obs):
        x = functional.relu(self.bn1(self.conv1(obs)))
        x = functional.relu(self.bn2(self.conv2(x)))
        x = functional.relu(self.bn3(self.conv3(x)))
        x = functional.relu(self.bn4(self.conv4(x)))
        x = torch.flatten(x, start_dim=1)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        return self.fc3(x)

