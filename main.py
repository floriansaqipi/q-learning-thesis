import gymnasium as gym

from q_learning_taxi_env_agent import Environment, VideoRecordingEnvironment, QLearningAgent, TrainingPlayer, \
    GraphVisualiser, RenderMode, StatisticsRecordingEnvironment

from q_learning_taxi_env_agent.constants import ENVIRONMENT_ID_TAXI_V3
from q_learning_taxi_env_agent.parameters import N_TRAINING_EPISODES, N_PLAYING_EPISODES, SEED, LEARNING_RATE, \
    START_EPSILON, EPSILON_DECAY, \
    FINAL_EPSILON, DISCOUNT_FACTOR

RECORD_FREQUENCY = N_TRAINING_EPISODES / 10

env = Environment(ENVIRONMENT_ID_TAXI_V3, RenderMode.RGB_ARRAY, SEED)
statistics_recording_env = StatisticsRecordingEnvironment(env, N_TRAINING_EPISODES)
video_recording_env = VideoRecordingEnvironment(statistics_recording_env, RECORD_FREQUENCY)

agent = QLearningAgent(video_recording_env, LEARNING_RATE, START_EPSILON, EPSILON_DECAY, FINAL_EPSILON, DISCOUNT_FACTOR)

player = TrainingPlayer(video_recording_env, N_TRAINING_EPISODES, agent)
player.play()
player.save_progress()


visualiser = GraphVisualiser(statistics_recording_env, agent)
visualiser.visualise()
