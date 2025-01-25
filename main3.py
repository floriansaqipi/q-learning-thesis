from q_learning_taxi_env_agent import Environment, VideoRecordingEnvironment, DeepQLearningAgent, TrainingDeepQLearningPlayer, \
    GraphVisualiser, RenderMode, StatisticsRecordingEnvironment


from q_learning_taxi_env_agent.agent import ReplayMemory

from q_learning_taxi_env_agent.constants import ENVIRONMENT_LUNAR_LANDER_V3, Q_NETWORK_WEIGHTS_PROGRESS_MEMORY_FILE_NAME, \
    DEEP_Q_LEARNING_TRAINING_PLAYER_VIDEOS_DIRECTORY_NAME
from q_learning_taxi_env_agent.hyper_parameters.deep_q_learning_hyper_parameters import N_TRAINING_EPISODES, SEED, LEARNING_RATE, \
    START_EPSILON, EPSILON_DECAY, \
    FINAL_EPSILON, DISCOUNT_FACTOR, INTERPOLATION_PARAMETER, REPLAY_MEMORY_SIZE, REPLAY_MEMORY_BATCH_SIZE

RECORD_FREQUENCY = N_TRAINING_EPISODES / 10

env = Environment(ENVIRONMENT_LUNAR_LANDER_V3, RenderMode.RGB_ARRAY, SEED)
statistics_recording_env = StatisticsRecordingEnvironment(env, N_TRAINING_EPISODES)
video_recording_env = VideoRecordingEnvironment(statistics_recording_env, RECORD_FREQUENCY, DEEP_Q_LEARNING_TRAINING_PLAYER_VIDEOS_DIRECTORY_NAME)

replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE, REPLAY_MEMORY_BATCH_SIZE)

agent = DeepQLearningAgent(video_recording_env, LEARNING_RATE, DISCOUNT_FACTOR, INTERPOLATION_PARAMETER, START_EPSILON, EPSILON_DECAY, FINAL_EPSILON, replay_memory)

player = TrainingDeepQLearningPlayer(video_recording_env, agent, N_TRAINING_EPISODES)
player.play()
player.save_progress(Q_NETWORK_WEIGHTS_PROGRESS_MEMORY_FILE_NAME)


visualiser = GraphVisualiser(statistics_recording_env, agent)
visualiser.visualise()
