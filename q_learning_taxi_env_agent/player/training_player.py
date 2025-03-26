import cProfile
import os
import time
from turtledemo.penrose import start

from fontTools.misc.plistlib import end_true

from ..agent import  Agent
from ..environment import Environment
from .player import Player


class TrainingPlayer(Player):
    def __init__(
            self,
            agent: Agent,
            env: Environment,
            n_episodes : int = None,
            save_frequency: int = None,
            printing_enabled: bool = False
    ):
        super().__init__(agent, env, n_episodes, printing_enabled)
        self.save_frequency = save_frequency

    def play(self):
        # profiler = cProfile.Profile()
        # profiler.enable()
        # # ... (existing play code)

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
            if self.printing_enabled:
                self.progress_bar.update(1)

            if self.save_frequency is not None and episode_count % self.save_frequency == 0:
                self.save_progress(self.save_frequency, episode_count)

        # profiler.disable()
        # profiler.dump_stats(f"profile_{os.getpid()}.prof")
        self.env.close()

