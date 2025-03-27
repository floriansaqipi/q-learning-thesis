import torch.multiprocessing as mp

from q_learning_taxi_env_agent import Environment, A3CConstants, RenderMode, StatisticsRecordingEnvironment, \
    VideoRecordingEnvironment, A3CAgent, LifeResetTrainingPlayer
from q_learning_taxi_env_agent.agent.A3C_experience_agent import A3CExperienceAgent
from q_learning_taxi_env_agent.agent.utils import A3CGlobalNetwork
from q_learning_taxi_env_agent.hyper_parameters import DECAY_FACTOR, EPSILON, VALUE_LOSS_COEFFICIENT
from q_learning_taxi_env_agent.hyper_parameters.A3C_hyper_parameters import N_TRAINING_EPISODES, N_WORKERS, \
    LEARNING_RATE, DISCOUNT_FACTOR, ENTROPY_REGULARIZATION_COEFFICIENT, N_STEPS, MAX_NORM
from q_learning_taxi_env_agent.result_visualiser.offline_graph_visualiser import OfflineGraphVisualiser

RECORD_FREQUENCY = N_TRAINING_EPISODES / 10
SAVING_FREQUENCY = N_TRAINING_EPISODES / 100


def create_main_worker():
    main_env = env
    statistics_recording_env = StatisticsRecordingEnvironment(main_env, N_TRAINING_EPISODES)
    video_recording_env = VideoRecordingEnvironment(statistics_recording_env, RECORD_FREQUENCY,
                                                    A3CConstants.TRAINING_PLAYER_VIDEOS_DIRECTORY_NAME)

    agent = A3CExperienceAgent(video_recording_env, a3c_global_network, DISCOUNT_FACTOR,
                               ENTROPY_REGULARIZATION_COEFFICIENT, VALUE_LOSS_COEFFICIENT, N_STEPS, MAX_NORM, True)

    player = LifeResetTrainingPlayer(agent, video_recording_env, N_TRAINING_EPISODES, SAVING_FREQUENCY, True)

    return player


def create_default_worker():
    default_env = Environment(A3CConstants.ENVIRONMENT_ALE_BREAKOUT_NO_FRAMESKIP_V4, RenderMode.RGB_ARRAY)

    agent = A3CAgent(default_env, a3c_global_network, DISCOUNT_FACTOR, ENTROPY_REGULARIZATION_COEFFICIENT,
                     VALUE_LOSS_COEFFICIENT, N_STEPS, MAX_NORM)

    player = LifeResetTrainingPlayer(agent, default_env, N_TRAINING_EPISODES, SAVING_FREQUENCY, True)

    return player


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    env = Environment(A3CConstants.ENVIRONMENT_ALE_BREAKOUT_NO_FRAMESKIP_V4, RenderMode.RGB_ARRAY)

    a3c_global_network = A3CGlobalNetwork(env, LEARNING_RATE, DECAY_FACTOR, EPSILON)
    a3c_global_network.load_progress()

    processes = []
    main_player = create_main_worker()
    process = mp.Process(target=main_player.play)
    process.start()
    processes.append(process)

    for i in range(N_WORKERS - 1):
        default_player = create_default_worker()
        process = mp.Process(target=default_player.play)
        process.start()
        processes.append(process)

    for p in processes:
        p.join()

    visualiser = OfflineGraphVisualiser(A3CConstants.PROGRESS_STATISTICS_FILE_NAME)
    visualiser.visualise()
