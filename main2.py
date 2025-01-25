

from q_learning_taxi_env_agent import Environment, QLearningAgent, RenderMode, StatisticsRecordingEnvironment, InferencePlayer

from q_learning_taxi_env_agent.constants import ENVIRONMENT_ID_TAXI_V3, PROGRESS_MEMORY_FILE_NAME
from q_learning_taxi_env_agent.hyper_parameters.parameters import N_TRAINING_EPISODES, SEED, LEARNING_RATE, \
    START_EPSILON, EPSILON_DECAY, \
    FINAL_EPSILON, DISCOUNT_FACTOR


env = Environment(ENVIRONMENT_ID_TAXI_V3, RenderMode.HUMAN, SEED)
statistics_recording_env = StatisticsRecordingEnvironment(env, N_TRAINING_EPISODES)

agent = QLearningAgent(statistics_recording_env, LEARNING_RATE, START_EPSILON, EPSILON_DECAY, FINAL_EPSILON, DISCOUNT_FACTOR)

player = InferencePlayer(statistics_recording_env, agent)
player.load_progress(PROGRESS_MEMORY_FILE_NAME)
player.play()



# visualiser = GraphVisualiser(statistics_recording_env, agent)
# visualiser.visualise()
