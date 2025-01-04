from abc import ABC, abstractmethod


class EpsilonGreedyPolicy(ABC):

    @abstractmethod
    def decay_epsilon(self):
        pass
