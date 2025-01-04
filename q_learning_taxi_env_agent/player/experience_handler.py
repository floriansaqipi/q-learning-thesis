
from abc import ABC, abstractmethod

class ExperienceHandler(ABC):

    @abstractmethod
    def load_progress(self):
        pass

    @abstractmethod
    def save_progress(self):
        pass