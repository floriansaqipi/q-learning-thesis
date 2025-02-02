from .constants import Constants

class ConvolutionalDeepQLearningConstants(Constants):
    PROGRESS_MEMORY_FILE_NAME = "convolutional_deep_q_network_weights.pth"
    TRAINING_PLAYER_VIDEOS_DIRECTORY_NAME = "training_convolutional_deep_q_learning_player_videos"
    INFERENCE_PLAYER_VIDEOS_DIRECTORY_NAME = "inference_convolutional_deep_q_learning_player_videos"
    PLAYER_NAME_PREFIX = "convolutional_deep_q_learning_player"