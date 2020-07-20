class BaseLoop:
    def __init__(self, **params):
        self.__dict__.update(self.get_default_parameters())
        self.__dict__.update(params)

    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    @staticmethod
    def get_default_parameters():
        raise NotImplementedError()
