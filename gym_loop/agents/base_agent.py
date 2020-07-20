from typing import Dict
import numpy as np


class BaseAgent:
    def __init__(self, **params: Dict):
        super().__init__()
        self.__dict__.update(self.get_default_parameters())
        self.__dict__.update(params)

    def act(self, state: np.ndarray, episode_num: int):
        """Retrieves agent's action upon state"""
        raise NotImplementedError()

    def memorize(
        self,
        last_ob: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        ob: np.ndarray,
    ):
        """Called after environment steps on action, arguments are classic SARSA tuple"""
        raise NotImplementedError()

    def update(self, episode_num: int):
        """Called immediately after memorize"""
        raise NotImplementedError()

    def metrics(self, episode_num: int) -> Dict:
        """Returns dict with metrics to log in tensorboard"""
        raise NotImplementedError()

    @staticmethod
    def get_default_parameters() -> Dict:
        """Specifies tweakable parameters for agents
        
        Returns:
            dict: default parameters for the agent
        """
        raise NotImplementedError()
