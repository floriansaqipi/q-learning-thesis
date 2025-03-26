import numpy as np

from .visualiser import Visualiser
from ..agent.utils.json_progress_handler import JsonProgressHandler
from ..constans import Constants


class OfflineGraphVisualiser(Visualiser):

    def __init__(self, file_name: str):
        super().__init__()
        self.data = JsonProgressHandler.load_statistics(file_name)

    def get_rewards(self):
        return self.data[Constants.STATISTICS_RETURN_QUEUE]

    def get_lengths(self):
        return self.data[Constants.STATISTICS_LENGTH_QUEUE]

    def get_training_error(self):
        return self.data[Constants.STATISTICS_TRAINING_ERROR]
