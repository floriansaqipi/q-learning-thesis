import gymnasium as gym

from .envrionment import Environment
from .envrionment_decorator import EnvironmentDecorator
from ..render_modes import RenderMode
from ..constants import PLAYER_VIDEOS_DIRECTORY, PLAYER_NAME_PREFIX


class VideoRecordingEnvironment(EnvironmentDecorator):
    def __init__(self, env: Environment, record_frequency: int):
        super().__init__(env)
        self.record_frequency = record_frequency
        self.env.inner_env = gym.wrappers.RecordVideo(
            self.env.inner_env,
            video_folder=PLAYER_VIDEOS_DIRECTORY, name_prefix=PLAYER_NAME_PREFIX,
            episode_trigger=lambda x : x % record_frequency == 0)