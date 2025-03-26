
from abc import ABC, abstractmethod

class ExperienceHandler(ABC):

    @abstractmethod
    def load_progress(self):
        pass

    @abstractmethod
    def save_progress(self, save_frequency: int, episode_number: int = 0, return_queue = None, length_queue = None):
        pass