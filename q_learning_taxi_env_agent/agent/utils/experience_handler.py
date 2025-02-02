
from abc import ABC, abstractmethod

class ExperienceHandler(ABC):

    @abstractmethod
    def load_progress(self, file_name: str):
        pass

    @abstractmethod
    def save_progress(self, file_name: str):
        pass