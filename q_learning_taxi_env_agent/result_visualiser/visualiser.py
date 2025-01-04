from abc import ABC, abstractmethod

from ..environment import StatisticsRecordingEnvironment
from ..agent import Agent

class Visualiser(ABC):
    def __init__(self, env: StatisticsRecordingEnvironment, agent: Agent):
        self.env = env
        self.agent = agent

    @abstractmethod
    def visualise(self):
        pass