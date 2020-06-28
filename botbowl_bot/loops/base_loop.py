class BaseLoop:
    @staticmethod
    def get_default_parameters():
        raise NotImplementedError()

    def __init__(self):
        super().__init__()
        self.parameters = self.get_default_parameters()
