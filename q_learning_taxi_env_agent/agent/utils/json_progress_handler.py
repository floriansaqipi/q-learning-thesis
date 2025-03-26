import json
import os.path

from ...constans import Constants

class JsonProgressHandler:

    @staticmethod
    def save_statistics(return_queue, length_queue, training_error,file_name: str):

        file_full_path = Constants.PROGRESS_MEMORY_DIRECTORY + file_name

        return_queue = [int(x) for x in return_queue]
        length_queue = [int(x) for x in length_queue]

        data = {
            Constants.STATISTICS_RETURN_QUEUE: return_queue,
            Constants.STATISTICS_LENGTH_QUEUE: length_queue,
            Constants.STATISTICS_TRAINING_ERROR: training_error
        }

        with open(file_full_path, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def save_episode_number(save_frequency: int, episode_number: int, file_name: str):
        file_full_path = Constants.PROGRESS_MEMORY_DIRECTORY + file_name

        current_episode = 0

        if os.path.exists(file_full_path):
            with open(file_full_path, 'r') as f:
                data = json.load(f)
                current_episode = data.get(Constants.STATISTICS_TOTAL_EPISODE_COUNT)

        updated_episode = current_episode + save_frequency

        data = {
            Constants.STATISTICS_TOTAL_EPISODE_COUNT: updated_episode,
            Constants.STATISTICS_LATEST_EPISODE_COUNT: episode_number
        }

        with open(file_full_path, 'w') as f:
            json.dump(data, f)

    @staticmethod
    def load_statistics(file_name: str):
        file_full_path = Constants.PROGRESS_MEMORY_DIRECTORY + file_name

        if not os.path.exists(file_full_path):
            return {
                Constants.STATISTICS_RETURN_QUEUE: [],
                Constants.STATISTICS_LENGTH_QUEUE: [],
                Constants.STATISTICS_TRAINING_ERROR: []
            }

        with open(file_full_path, 'r') as f:
            data = json.load(f)

        return data

    @staticmethod
    def load_episode_number(file_name: str):
        file_full_path = Constants.PROGRESS_MEMORY_DIRECTORY + file_name

        if not os.path.exists(file_full_path):
            return

        with open(file_full_path, 'r') as f:
            data = json.load(f)

        return data