class BaseAgent:
    def __init__(self, **params):
        super().__init__()
        self.__dict__.update(self.get_default_parameters())
        self.__dict__.update(params)

    def act(self, state, episode_num):
        """Retrieves agent's action upon state"""
        raise NotImplementedError()

    def memorize(self, last_ob, action, reward, done, ob):
        """Called after environment steps on action, arguments are classic SARSA tuple"""
        raise NotImplementedError()

    def update(self, episode_num):
        """Called immediately after memorize"""
        raise NotImplementedError()

    @staticmethod
    def get_default_parameters():
        """Specifies tweakable parameters for agents
        
        Returns:
            dict: default parameters for the agent
        """
        raise NotImplementedError()
