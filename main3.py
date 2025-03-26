from q_learning_taxi_env_agent import Environment, VideoRecordingEnvironment, DeepQLearningAgent, TrainingPlayer, \
    GraphVisualiser, RenderMode, StatisticsRecordingEnvironment, DeepQLearningConstants, ReplayMemory

from q_learning_taxi_env_agent.hyper_parameters.deep_q_learning_hyper_parameters import \
    N_TRAINING_EPISODES, LEARNING_RATE, \
    START_EPSILON, EPSILON_DECAY, \
    FINAL_EPSILON, DISCOUNT_FACTOR, INTERPOLATION_PARAMETER, REPLAY_MEMORY_SIZE, REPLAY_MEMORY_BATCH_SIZE

RECORD_FREQUENCY = N_TRAINING_EPISODES / 10

env = Environment(DeepQLearningConstants.ENVIRONMENT_LUNAR_LANDER_V3, RenderMode.RGB_ARRAY)
statistics_recording_env = StatisticsRecordingEnvironment(env, N_TRAINING_EPISODES)
video_recording_env = VideoRecordingEnvironment(statistics_recording_env, RECORD_FREQUENCY,
                                                DeepQLearningConstants.TRAINING_PLAYER_VIDEOS_DIRECTORY_NAME)

replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE, REPLAY_MEMORY_BATCH_SIZE)

agent = DeepQLearningAgent(video_recording_env, LEARNING_RATE, DISCOUNT_FACTOR, INTERPOLATION_PARAMETER, START_EPSILON,
                           EPSILON_DECAY, FINAL_EPSILON, replay_memory)

player = TrainingPlayer(agent, video_recording_env, N_TRAINING_EPISODES)
player.play()
player.save_progress(0)

visualiser = GraphVisualiser(statistics_recording_env, agent)
visualiser.visualise()
