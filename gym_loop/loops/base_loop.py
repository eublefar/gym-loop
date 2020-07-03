class BaseLoop:
    def __init__(self):
        super().__init__()
        self.parameters = self.get_default_parameters()

    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    @staticmethod
    def get_default_parameters():
        raise NotImplementedError()
