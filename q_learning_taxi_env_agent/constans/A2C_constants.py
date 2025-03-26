from .constants import Constants

class A2CConstants(Constants):
    PROGRESS_MEMORY_FILE_NAME = "A2C_network_checkpoint.pth"
    PROGRESS_STATISTICS_FILE_NAME = "A2C_statistics_checkpoint.json"
    PROGRESS_EPISODE_NUMBER_FILE_NAME = "A2C_episode_number.json"
    TRAINING_PLAYER_VIDEOS_DIRECTORY_NAME = "training_A2C_player_videos"
    INFERENCE_PLAYER_VIDEOS_DIRECTORY_NAME = "inference_A2C_player_videos"
    PLAYER_NAME_PREFIX = "A2C_learning_player"
