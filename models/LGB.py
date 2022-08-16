from models import Hyperopt
from lightgbm import LGBMClassifier


class LGB(Hyperopt):
    def __init__(self, cfg):
        super(LGB, self).__init__(cfg, LGBMClassifier)
