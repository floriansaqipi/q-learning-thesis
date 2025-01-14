

from q_learning_taxi_env_agent import Environment, VideoRecordingEnvironment, QLearningAgent, \
    GraphVisualiser, RenderMode, StatisticsRecordingEnvironment, InferencePlayer

from q_learning_taxi_env_agent.constants import ENVIRONMENT_ID_TAXI_V3, PROGRESS_MEMORY_FILE_NAME, \
     INFERENCE_PLAYER_VIDEOS_DIRECTORY_NAME
from q_learning_taxi_env_agent.parameters import N_TRAINING_EPISODES, N_PLAYING_EPISODES, SEED, LEARNING_RATE, \
    START_EPSILON, EPSILON_DECAY, \
    FINAL_EPSILON, DISCOUNT_FACTOR

RECORD_FREQUENCY = N_PLAYING_EPISODES / 10

env = Environment(ENVIRONMENT_ID_TAXI_V3, RenderMode.RGB_ARRAY, SEED)
statistics_recording_env = StatisticsRecordingEnvironment(env, N_TRAINING_EPISODES)
video_recording_env = VideoRecordingEnvironment(statistics_recording_env, RECORD_FREQUENCY, INFERENCE_PLAYER_VIDEOS_DIRECTORY_NAME)

agent = QLearningAgent(video_recording_env, LEARNING_RATE, START_EPSILON, EPSILON_DECAY, FINAL_EPSILON, DISCOUNT_FACTOR)

player = InferencePlayer(video_recording_env, agent, N_PLAYING_EPISODES)
player.load_progress(PROGRESS_MEMORY_FILE_NAME)
player.play()



visualiser = GraphVisualiser(statistics_recording_env, agent)
visualiser.visualise()
