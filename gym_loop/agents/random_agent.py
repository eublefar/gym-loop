from .base_agent import BaseAgent
import numpy as np


class RandomAgent(BaseAgent):
    @staticmethod
    def get_default_parameters():
        return {}

    def __init__(self, **params):
        super().__init__(**params)
        self.action_space = params["action_space"]

    def memorize(
        self,
        last_ob: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        ob: np.ndarray,
        global_step: int,
    ):
        pass

    def act(self, state, episode_num):
        return self.action_space.sample()

    def update(self, episode_num):
        pass

    def metrics(self, episode_num):
        return {"step": episode_num}
