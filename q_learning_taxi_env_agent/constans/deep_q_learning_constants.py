from .constants import Constants

class DeepQLearningConstants(Constants):
    PROGRESS_MEMORY_FILE_NAME = "q_network_weights.pth"
    TRAINING_PLAYER_VIDEOS_DIRECTORY_NAME = "training_deep_q_learning_player_videos"
    INFERENCE_PLAYER_VIDEOS_DIRECTORY_NAME = "inference_deep_q_learning_player_videos"
    PLAYER_NAME_PREFIX = "deep_q_learning_player"