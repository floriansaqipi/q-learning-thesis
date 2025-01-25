import torch
import torch.nn as nn
import torch.nn.functional as functional

from q_learning_taxi_env_agent.hyper_parameters.parameters import SEED


class DeepQLearningNetwork(nn.Module):
    def __init__(self, observation_space_size: int, action_space_size: int, seed: int = SEED):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size

        self.fc1 = nn.Linear(self.observation_space_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.action_space_size)

    def forward(self, obs):
        x = self.fc1(obs)
        x = functional.relu(x)
        x = self.fc2(x)
        x = functional.relu(x)
        return self.fc3(x)