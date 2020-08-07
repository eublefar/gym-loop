from typing import Union, Dict, Any
import numpy as np


class BasePolicy:
    def __init__(self, observation_space, action_space):
        pass

    def act(self, state: Union[np.ndarray, Dict, Any]) -> np.ndarray:
        """Act on state"""
        raise NotImplementedError()

    def batch_act(self, state: Union[np.ndarray, Dict, Any]) -> np.ndarray:
        """Act on batch of state"""
        raise NotImplementedError()

    def __call__(self, state_batch: Union[np.ndarray, Dict, Any]) -> Dict[str, Any]:
        """Batch prediction"""
        raise NotImplementedError()

    def load(self, path: str):
        """Initialize policy from path"""
        raise NotImplementedError

    def save(self, path: str):
        """Save policy to path"""
        raise NotImplementedError

    @staticmethod
    def get_default_parameters():
        return {}
