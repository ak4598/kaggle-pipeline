from models import Hyperopt
from lightgbm import LGBMClassifier


class LGB_Hyperopt(Hyperopt):
    def __init__(self, cfg):
        super(LGB_Hyperopt, self).__init__(cfg, LGBMClassifier)
