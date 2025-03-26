from collections import deque

from ..agent import  Agent
from ..environment import Environment
from .training_player import TrainingPlayer

class VectorizedTrainingPlayer(TrainingPlayer):

    def __init__(self, agent: Agent, env: Environment, n_episodes : int = None, save_frequency: int = None):
        super().__init__(agent, env, n_episodes, save_frequency)
        self.rewards_queue = deque(maxlen=self.n_episodes)
        self.length_queue = deque(maxlen=self.n_episodes)


    def play(self):
        episode_count = 0
        obs, info = self.env.reset(self.env.seed)

        while self.n_episodes is None or episode_count < self.n_episodes:
            done  = False

            while not done:
                action = self.agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.agent.update(obs, action, reward, terminated, next_obs)
                obs = next_obs
                done = terminated[0] or truncated[0]

            self.rewards_queue.append(info["episode"]["r"][0])
            self.length_queue.append(info["episode"]["l"][0])

            episode_count += 1
            self.progress_bar.update(1)

            if self.save_frequency is not None and episode_count % self.save_frequency == 0:
                self.save_progress(self.save_frequency, episode_count, self.rewards_queue, self.length_queue)

        self.env.close()