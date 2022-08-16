import os
import pickle
from re import search
import numpy as np
import pandas as pd
from metric import amex_metric
from pathlib import Path
from datetime import datetime
from interfaces import IModel
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer


class LGB(IModel):
    def __init__(self, cfg):
        super(LGB, self).__init__(cfg)
        self.amex_scorer = make_scorer(amex_metric, needs_proba=True)
        self.score = 0

    def build(self):
        print("Building model...")

        if os.path.isfile(self.cfg["train"]["load_from"]):
            print("Loading existing model: %s" %
                  self.cfg["train"]["load_from"])
            self.model = pickle.load(
                open(self.cfg["train"]["load_from"], 'rb'))

        else:
            self.model = LGBMClassifier(**self.cfg["clf"]["init"])

    def train(self, X_train, Y_train):
        assert (isinstance(self.model, LGBMClassifier)
                ), "LGB model is not yet built!"

        if not self.cfg["train"]["train"]:
            print("You chose to skip training...")
            return

        self.X_train = X_train
        self.Y_train = Y_train

        # Load params that are indicated to be searched in the cfg
        search_cfg = self.cfg["clf"]["searching_space"]
        params_to_search = [
            param for param in search_cfg if search_cfg[param]["search"] == True]

        self.param = {
            key: hp.quniform(key, *search_cfg[key]["space"].values())
            if "q" in search_cfg[key]["space"].keys()
            else hp.uniform(key, *search_cfg[key]["space"].values())
            for key in params_to_search}

        print("Start training...")
        best = fmin(fn=self.objective,
                    space=self.param,
                    algo=tpe.suggest,
                    max_evals=self.cfg["train"]["itr"]
                    )

        # exclude params that are not in the searching space
        params_not_to_search = [
            param for param in self.cfg["clf"]["init"].keys() if param not in params_to_search]

        # cast to int/float based on the params' types at init stage
        final_model = LGBMClassifier(
            **{param: self.cfg["clf"]["init"][param] for param in params_not_to_search},
            **{param: int(best[param]) if isinstance(self.cfg["clf"]["init"][param], int) else best[param] for param in best.keys()}
        )

        final_model.fit(self.X_train, self.Y_train)
        self.model = final_model

    def eval(self, X_test, Y_test):
        assert (isinstance(self.model, LGBMClassifier)
                ), "LGB model is not yet built!"

        print("Evaluating model...")
        Y_pred = self.model.predict_proba(X_test)[:, 1]

        self.score = -amex_metric(Y_test, Y_pred)

        print("Score: %.6f" % (self.score))

        if self.cfg["train"]["save"]:
            self.save_output()

    ##############
    ####utils#####
    ##############

    def objective(self, packed_inputs):
        score = cross_val_score(self.model,
                                self.X_train,
                                self.Y_train,
                                scoring=self.amex_scorer,
                                cv=self.cfg["clf"]["cross_validation"],
                                n_jobs=self.cfg["clf"]["init"]["n_jobs"]
                                ).mean()

        print("CV score = {:.3f}, params = {}".format(-score, packed_inputs))
        return score

    def save_output(self):
        print('Saving best model...')

        dt = datetime.now().strftime('%Y%m%d%H%M%S')

        final_score_str = "%.6f" % (self.score)

        dataset_path = Path(__file__).parent.parent.parent.resolve() / \
            'data' / 'Amex' / 'dataset' / 'output'

        output_dir = "LGB_%s_%s" % (final_score_str, dt)
        output_path = Path(__file__).parent.parent.parent.resolve() / \
            'output' / 'Amex' / output_dir

        os.mkdir(output_path)

        model_sav = "LGB_%s_%s.sav" % (final_score_str, dt)
        pickle.dump(self.model, open(output_path / model_sav, 'wb'))

        print("Best model saved.")

        print('Reading kaggle test data for prediction...')
        kaggle_test = pd.read_parquet(
            dataset_path / 'test_data_reduced.parquet')

        print('Predicting output...')
        submit_df = kaggle_test[['customer_ID']]

        submit_df['prediction'] = self.model.predict_proba(
            kaggle_test.drop(['customer_ID'], axis=1))[:, 1]

        print('Saving output to csv...')
        submission_csv = "LGB_%s_%s.csv" % (final_score_str, dt)
        submit_df.to_csv(
            output_path / submission_csv, index=False)

        print("Saved.")
