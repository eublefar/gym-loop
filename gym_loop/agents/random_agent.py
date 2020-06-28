from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    @staticmethod
    def get_default_parameters():
        return {}

    def __init__(self, **params):
        super().__init__(**params)
        self.action_space = params["action_space"]

    def memorize(self, last_ob, action, reward, done, ob):
        pass

    def act(self, state, episode_num):
        return self.action_space.sample()

    def update(self, episode_num):
        pass
