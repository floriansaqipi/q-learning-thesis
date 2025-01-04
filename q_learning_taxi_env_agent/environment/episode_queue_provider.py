from abc import ABC,abstractmethod


class EpisodeQueueProvider(ABC):

    @abstractmethod
    def get_return_queue(self):
        pass

    @abstractmethod
    def get_length_queue(self):
        pass