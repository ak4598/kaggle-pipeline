from models import Hyperopt
from catboost import CatBoostClassifier


class CAT_Hyperopt(Hyperopt):
    def __init__(self, cfg):
        super(CAT_Hyperopt, self).__init__(cfg, CatBoostClassifier)
