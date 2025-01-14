
import gymnasium as gym

from .envrionment import Environment
from .environment_wrapper import EnvironmentWrapper
from ..constants import PLAYER_VIDEOS_DIRECTORY, PLAYER_NAME_PREFIX


class VideoRecordingEnvironment(EnvironmentWrapper):
    def __init__(self, env: Environment, record_frequency: int, video_directory_name: str):
        super().__init__(env)
        self.record_frequency = record_frequency
        self.video_directory_name = video_directory_name
        self.inner_env = gym.wrappers.RecordVideo(
            self.env.inner_env,
            video_folder=PLAYER_VIDEOS_DIRECTORY + self.video_directory_name, name_prefix=PLAYER_NAME_PREFIX,
            episode_trigger=lambda x : x % record_frequency == 0)