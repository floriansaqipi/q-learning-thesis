

from q_learning_taxi_env_agent import Environment, QLearningAgent, RenderMode, StatisticsRecordingEnvironment, InferencePlayer, \
    QLearningConstants

from q_learning_taxi_env_agent.hyper_parameters.q_learning_hyper_parameters import N_TRAINING_EPISODES, SEED, LEARNING_RATE, \
    START_EPSILON, EPSILON_DECAY, \
    FINAL_EPSILON, DISCOUNT_FACTOR


env = Environment(QLearningConstants.ENVIRONMENT_ID_TAXI_V3, RenderMode.HUMAN, SEED)
statistics_recording_env = StatisticsRecordingEnvironment(env, N_TRAINING_EPISODES)

agent = QLearningAgent(statistics_recording_env, LEARNING_RATE, START_EPSILON, EPSILON_DECAY, FINAL_EPSILON, DISCOUNT_FACTOR)

player = InferencePlayer(agent, statistics_recording_env)
player.load_progress(QLearningConstants.PROGRESS_MEMORY_FILE_NAME)
player.play()



# visualiser = GraphVisualiser(statistics_recording_env, agent)
# visualiser.visualise()
