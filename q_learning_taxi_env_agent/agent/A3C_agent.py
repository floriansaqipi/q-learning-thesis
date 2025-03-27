import numpy as np
import torch
import torch.nn.functional as functional
from torch.xpu import device

from .agent import Agent
from .network import A3CNetwork
from .utils import ComputeDevice, FramePreprocessor, TransitionConverter, A3CGlobalNetwork, JsonProgressHandler

from ..environment import Environment, EnvironmentWrapper, StatisticsRecordingEnvironment



class A3CAgent(Agent):
    def __init__(
            self,
            env: Environment,
            a3c_global_network: A3CGlobalNetwork,
            discount_factor: float,
            entropy_regularization_coefficient: float,
            value_loss_coefficient: float,
            n_steps: int,
            max_norm: float,
            training_error_enabled: bool = False
    ):
        super().__init__(env)
        self.a3c_global_network = a3c_global_network
        self.discount_factor = discount_factor
        self.entropy_regularization_coefficient = entropy_regularization_coefficient
        self.value_loss_coefficient = value_loss_coefficient
        self.n_steps = n_steps
        self.max_norm = max_norm
        self.training_error_enabled = training_error_enabled
        self.n_step_transitions = []
        self.device = ComputeDevice.get_device()
        self.action_space = self.env.get_action_space().n
        self.network = A3CNetwork(self.action_space).to(device=self.device)
        self.network.load_state_dict(self.a3c_global_network.network.state_dict())
        self.frame_preprocessor = FramePreprocessor()
        self.transition_converter = TransitionConverter()


    def get_action(self, obs):
        obs = self.frame_preprocessor.preprocess_frame(obs)
        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        self.network.eval()
        with torch.no_grad():
            action_values, _ = self.network(obs)
        self.network.train()

        policy = functional.softmax(action_values, dim=-1)
        action_indices = torch.multinomial(policy, 1).squeeze(0)
        return action_indices.item()


    def get_best_action(self, obs):
        obs = self.frame_preprocessor.preprocess_frame(obs)
        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        self.network.eval()
        with torch.no_grad():
            action_values, _ = self.network(obs)
        self.network.train()

        best_action_indices = torch.argmax(action_values, dim=1).squeeze()

        return  best_action_indices.item()

    def update(self, obs, action, reward, terminated, next_obs, truncated=None):
        obs = self.frame_preprocessor.preprocess_frame(obs)
        next_obs = self.frame_preprocessor.preprocess_frame(next_obs)
        self.n_step_transitions.append((obs, action, reward, terminated, next_obs))
        if len(self.n_step_transitions) % self.n_steps == 0 or terminated or truncated:
            obs_batch, action_batch, reward_batch, terminated_batch, next_obs_batch = (
                self.transition_converter.to_tensor_zip(self.n_step_transitions)
            )

            self.learn(obs_batch, action_batch, reward_batch, terminated_batch, next_obs_batch)
            self.n_step_transitions.clear()
            self.network.load_state_dict(self.a3c_global_network.network.state_dict())

    def learn(self, obs_batch, action_batch, reward_batch, terminated_batch, next_obs_batch):

        policy_loss = 0.0
        value_loss = 0.0
        entropy_loss = 0.0

        returns = []

        if terminated_batch[-1]:
            R = 0.0
        else:
            with torch.no_grad():
                _, next_value = self.network(next_obs_batch[-1].unsqueeze(0))
            R = next_value.item()

        reversed_rewards = torch.flip(reward_batch, dims=[0])

        for reward in reversed_rewards:
            R = reward.item() + self.discount_factor * R
            returns.insert(0, R)

        returns = torch.tensor(np.array(returns), dtype=torch.float32, device=self.device)

        for obs, action, reward, terminated, next_obs, R in zip(obs_batch, action_batch, reward_batch, terminated_batch, next_obs_batch, returns):
            logits, value = self.network(obs.unsqueeze(0))
            advantage = R - value.item()

            log_prob = functional.log_softmax(logits, dim=-1)[0, action]
            policy_loss += -log_prob * advantage
            value_loss += functional.mse_loss(value, R.unsqueeze(0))
            entropy_loss += -(functional.softmax(logits, dim=-1) * functional.log_softmax(logits, dim=-1)).sum()

        total_loss = policy_loss + self.value_loss_coefficient * value_loss - self.entropy_regularization_coefficient * entropy_loss

        if self.training_error_enabled :
            self.training_error.append(total_loss.detach().item())

        with torch.multiprocessing.Lock():
            self.a3c_global_network.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_norm)
            self.update_global()
            self.a3c_global_network.optimizer.step()


    def update_global(self):
        for local_param, global_param in zip(self.network.parameters(), self.a3c_global_network.network.parameters()):
            if global_param.grad is None:
                global_param._grad = local_param.grad
            else:
                global_param._grad += local_param.grad


    def load_progress(self):
        pass

    def save_progress(self, save_frequency: int, episode_number: int = 0, return_queue=None, length_queue=None):
        pass