from abc import ABC, abstractmethod


class IData(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def clean(self):
        pass

    @abstractmethod
    def split(self):
        pass

    @abstractmethod
    def save(self):
        pass
