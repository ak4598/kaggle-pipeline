from abc import ABC, abstractmethod


class IModel(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass
