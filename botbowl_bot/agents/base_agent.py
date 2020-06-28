class BaseAgent:
    @staticmethod
    def get_default_parameters():
        raise NotImplementedError()

    def act(self):
        raise NotImplementedError()

    def memorize(self):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()

    def __init__(self):
        super().__init__()
        self.parameters = self.get_default_parameters()
