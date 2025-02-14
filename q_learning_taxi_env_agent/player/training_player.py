
from ..agent import QLearningAgent, Agent
from ..environment import Environment
from .player import Player


class TrainingPlayer(Player):
    def __init__(self, agent: Agent, env: Environment, n_episodes : int = None):
        super().__init__(agent, env, n_episodes)

    def play(self):
        episode_count = 0

        while self.n_episodes is None or episode_count < self.n_episodes:
            obs, info = self.env.reset(self.env.seed)
            done = False

            while not done:
                action = self.agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.agent.update(obs, action, reward, terminated, next_obs)
                obs = next_obs
                done = terminated or truncated

            self.agent.end_of_episode_hook()
            episode_count += 1
            self.progress_bar.update(1)

        self.env.close()

