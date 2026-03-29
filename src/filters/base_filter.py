

class BaseFilter:
    def __init__(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, measurement):
        raise NotImplementedError




