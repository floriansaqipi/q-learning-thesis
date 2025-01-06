from abc import ABC
from typing import override

from .envrionment import Environment


class EnvironmentWrapper(Environment, ABC):
    def __init__(self, env: Environment):
        super().__init__(env.env_id, env.render_mode, env.seed)
        self.env = env



