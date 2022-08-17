from models import Hyperopt
from xgboost import XGBClassifier


class XGB_Hyperopt(Hyperopt):
    def __init__(self, cfg):
        super(XGB_Hyperopt, self).__init__(cfg, XGBClassifier)
