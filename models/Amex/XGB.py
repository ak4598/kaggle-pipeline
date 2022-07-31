import os
import pickle
import numpy as np
import pandas as pd
from metric import amex_metric
from pathlib import Path
from datetime import datetime
from interfaces import IModel
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer


class XGB(IModel):
    def __init__(self, cfg):
        super(XGB, self).__init__(cfg)
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
            self.param = {
                "n_estimators": hp.quniform("n_estimators",
                                            self.cfg["clf"]["search"]["n_estimators"]["low"],
                                            self.cfg["clf"]["search"]["n_estimators"]["high"],
                                            self.cfg["clf"]["search"]["n_estimators"]["q"]),
                "max_depth": hp.quniform("max_depth",
                                         self.cfg["clf"]["search"]["max_depth"]["low"],
                                         self.cfg["clf"]["search"]["max_depth"]["high"],
                                         self.cfg["clf"]["search"]["max_depth"]["q"]),
            }

            self.model = XGBClassifier(objective=self.cfg["clf"]["init"]["objective"],
                                       tree_method=self.cfg["clf"]["init"]["tree_method"],
                                       booster=self.cfg["clf"]["init"]["booster"],
                                       use_label_encoder=self.cfg["clf"]["init"]["use_label_encoder"],
                                       learning_rate=self.cfg["clf"]["init"]["learning_rate"],
                                       n_jobs=self.cfg["clf"]["init"]["n_jobs"],
                                       n_estimators=self.param["n_estimators"],
                                       max_depth=self.param["max_depth"])

    def train(self, X_train, Y_train):
        assert(isinstance(self.model, XGBClassifier)
               ), "XGB model is not yet built!"

        if not self.cfg["train"]["train"]:
            print("You chose to skip training...")
            return

        self.X_train = X_train
        self.Y_train = Y_train

        print("Start training...")
        best = fmin(fn=self.objective,
                    space=self.param,
                    algo=tpe.suggest,
                    max_evals=self.cfg["train"]["itr"]
                    )

        final_model = XGBClassifier(objective=self.cfg["clf"]["init"]["objective"],
                                    tree_method=self.cfg["clf"]["init"]["tree_method"],
                                    booster=self.cfg["clf"]["init"]["booster"],
                                    use_label_encoder=self.cfg["clf"]["init"]["use_label_encoder"],
                                    learning_rate=self.cfg["clf"]["init"]["learning_rate"],
                                    n_jobs=self.cfg["clf"]["init"]["n_jobs"],
                                    n_estimators=int(best["n_estimators"]),
                                    max_depth=int(best["max_depth"]))

        final_model.fit(self.X_train.drop(
            ['customer_ID'], axis=1), np.ravel(self.Y_train))
        self.model = final_model

    def eval(self, X_test, Y_test):
        assert(isinstance(self.model, XGBClassifier)
               ), "XGB model is not yet built!"

        print("Evaluating model...")
        Y_pred = self.model.predict_proba(
            X_test.drop(['customer_ID'], axis=1))[:, 1]

        self.score = -amex_metric(Y_test.to_numpy(), Y_pred)

        print("Score: %.6f" % (self.score))

        if self.cfg["train"]["save"]:
            self.save_output()

    ##############
    ####utils#####
    ##############

    def objective(self, packed_inputs):
        self.model = XGBClassifier(objective=self.cfg["clf"]["init"]["objective"],
                                   tree_method=self.cfg["clf"]["init"]["tree_method"],
                                   booster=self.cfg["clf"]["init"]["booster"],
                                   use_label_encoder=self.cfg["clf"]["init"]["use_label_encoder"],
                                   learning_rate=self.cfg["clf"]["init"]["learning_rate"],
                                   n_jobs=self.cfg["clf"]["init"]["n_jobs"],
                                   n_estimators=int(
                                       packed_inputs["n_estimators"]),
                                   max_depth=int(packed_inputs["max_depth"]))

        score = cross_val_score(self.model,
                                self.X_train.drop(['customer_ID'], axis=1),
                                np.ravel(self.Y_train),
                                scoring=self.amex_scorer,
                                cv=self.cfg["clf"]["search"]["cv"],
                                n_jobs=self.cfg["clf"]["init"]["n_jobs"]
                                ).mean()

        print("Amex CV score = {:.3f}, params = {}".format(-score, self.param))
        return score

    def save_output(self):
        print('Saving best model...')

        dt = datetime.now().strftime('%Y%m%d%H%M%S')

        final_score_str = "%.6f" % (self.score)

        dataset_path = Path(__file__).parent.parent.parent.resolve() / \
            'data' / 'Amex' / 'dataset' / 'output'

        output_dir = "XGB_%s_%s" % (final_score_str, dt)
        output_path = Path(__file__).parent.parent.parent.resolve() / \
            'output' / 'Amex' / output_dir

        os.mkdir(output_path)

        model_sav = "XGB_%s_%s.sav" % (final_score_str, dt)
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
        submission_csv = "XGB_%s_%s.csv" % (final_score_str, dt)
        submit_df.to_csv(
            output_path / submission_csv, index=False)

        print("Saved.")
