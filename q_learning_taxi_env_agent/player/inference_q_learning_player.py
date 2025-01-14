
from .q_learning_player import QLearningPlayer
from .. import StatisticsRecordingEnvironment
from ..agent import QLearningAgent
from ..environment import Environment



class InferencePlayer(QLearningPlayer):
    def __init__(self,env: Environment, agent: QLearningAgent, n_episodes: int = None):
        super().__init__(env, agent, n_episodes)

    def play(self):
        episode_count = 0

        while self.n_episodes is None or episode_count < self.n_episodes:
            obs, info = self.env.reset(self.env.seed)
            done = False

            while not done:
                action = self.agent.get_best_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                obs = next_obs
                done = terminated or truncated

            episode_count += 1
            self.progress_bar.update(1)

        self.env.close()
