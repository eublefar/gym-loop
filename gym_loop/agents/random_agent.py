from .base_agent import BaseAgent
import numpy as np


class RandomAgent(BaseAgent):
    @staticmethod
    def get_default_parameters():
        return {}

    def __init__(self, **params):
        super().__init__(**params)
        self.action_space = params["action_space"]
        self.sample = self.action_space.sample()

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

    def batch_act(self, state_batch, done_mask):
        actions = np.stack([np.zeros_like(self.sample)] * done_mask.shape[0])
        sampled = np.stack([self.action_space.sample()] * sum(done_mask))
        actions[done_mask == 1] = sampled
        return actions

    def batch_memorize(self, batch_transitions):
        print(batch_transitions)
        pass

    def update(self, episode_num):
        pass

    def metrics(self, episode_num):
        return {"step": episode_num}
