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
            self.param = {
                "n_estimators": hp.quniform("n_estimators",
                                            self.cfg["clf"]["search"]["n_estimators"]["low"],
                                            self.cfg["clf"]["search"]["n_estimators"]["high"],
                                            self.cfg["clf"]["search"]["n_estimators"]["q"]),
                "max_depth": hp.quniform("max_depth",
                                         self.cfg["clf"]["search"]["max_depth"]["low"],
                                         self.cfg["clf"]["search"]["max_depth"]["high"],
                                         self.cfg["clf"]["search"]["max_depth"]["q"]),

                "num_leaves": hp.quniform("num_leaves",
                                          self.cfg["clf"]["search"]["num_leaves"]["low"],
                                          self.cfg["clf"]["search"]["num_leaves"]["high"],
                                          self.cfg["clf"]["search"]["num_leaves"]["q"]),

                "min_split_gain": hp.uniform("min_split_gain",
                                             self.cfg["clf"]["search"]["min_split_gain"]["low"],
                                             self.cfg["clf"]["search"]["min_split_gain"]["high"]),

                "min_child_weight": hp.uniform("min_child_weight",
                                               self.cfg["clf"]["search"]["min_child_weight"]["low"],
                                               self.cfg["clf"]["search"]["min_child_weight"]["high"]),

                "subsample_for_bin": hp.quniform("subsample_for_bin",
                                                 self.cfg["clf"]["search"]["subsample_for_bin"]["low"],
                                                 self.cfg["clf"]["search"]["subsample_for_bin"]["high"],
                                                 self.cfg["clf"]["search"]["subsample_for_bin"]["q"]),

                "reg_alpha": hp.quniform("reg_alpha",
                                         self.cfg["clf"]["search"]["reg_alpha"]["low"],
                                         self.cfg["clf"]["search"]["reg_alpha"]["high"],
                                         self.cfg["clf"]["search"]["reg_alpha"]["q"]),

                "reg_lambda": hp.quniform("reg_lambda",
                                          self.cfg["clf"]["search"]["reg_lambda"]["low"],
                                          self.cfg["clf"]["search"]["reg_lambda"]["high"],
                                          self.cfg["clf"]["search"]["reg_lambda"]["q"]),

                "learning_rate": hp.uniform("learning_rate",
                                            self.cfg["clf"]["search"]["learning_rate"]["low"],
                                            self.cfg["clf"]["search"]["learning_rate"]["high"])
            }

            self.model = LGBMClassifier(boosting_type=self.cfg["clf"]["init"]["boosting_type"],
                                        class_weight=self.cfg["clf"]["init"]["class_weight"],
                                        objective=self.cfg["clf"]["init"]["objective"],
                                        xgboost_dart_mode=self.cfg["clf"]["init"]["xgboost_dart_mode"],
                                        n_jobs=self.cfg["clf"]["init"]["n_jobs"],
                                        n_estimators=self.param["n_estimators"],
                                        max_depth=self.param["max_depth"],
                                        num_leaves=self.param["num_leaves"],
                                        min_split_gain=self.param["min_split_gain"],
                                        min_child_weight=self.param["min_child_weight"],
                                        subsample_for_bin=self.param["subsample_for_bin"],
                                        reg_alpha=self.param["reg_alpha"],
                                        reg_lambda=self.param["reg_lambda"],
                                        learning_rate=self.param["learning_rate"])

    def train(self, X_train, Y_train):
        assert(isinstance(self.model, LGBMClassifier)
               ), "LGB model is not yet built!"

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

        final_model = LGBMClassifier(boosting_type=self.cfg["clf"]["init"]["boosting_type"],
                                     class_weight=self.cfg["clf"]["init"]["class_weight"],
                                     objective=self.cfg["clf"]["init"]["objective"],
                                     xgboost_dart_mode=self.cfg["clf"]["init"]["xgboost_dart_mode"],
                                     n_jobs=self.cfg["clf"]["init"]["n_jobs"],
                                     n_estimators=int(best["n_estimators"]),
                                     max_depth=int(best["max_depth"]),
                                     num_leaves=int(best["num_leaves"]),
                                     min_split_gain="{:.3f}".format(
                                         best["min_split_gain"]),
                                     min_child_weight="{:.3f}".format(
                                         best["min_child_weight"]),
                                     subsample_for_bin=int(
                                         best["subsample_for_bin"]),
                                     reg_alpha="{:.3f}".format(
                                         best["reg_alpha"]),
                                     reg_lambda="{:.3f}".format(
                                         best["reg_lambda"]),
                                     learning_rate="{:.3f}".format(best["learning_rate"]))

        final_model.fit(self.X_train.drop(
            ['customer_ID'], axis=1), np.ravel(self.Y_train))
        self.model = final_model

    def eval(self, X_test, Y_test):
        assert(isinstance(self.model, LGBMClassifier)
               ), "LGB model is not yet built!"

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
        self.model = LGBMClassifier(boosting_type=self.cfg["clf"]["init"]["boosting_type"],
                                    class_weight=self.cfg["clf"]["init"]["class_weight"],
                                    objective=self.cfg["clf"]["init"]["objective"],
                                    xgboost_dart_mode=self.cfg["clf"]["init"]["xgboost_dart_mode"],
                                    n_jobs=self.cfg["clf"]["init"]["n_jobs"],
                                    n_estimators=int(
                                        packed_inputs["n_estimators"]),
                                    max_depth=int(packed_inputs["max_depth"]),
                                    num_leaves=int(
                                        packed_inputs["num_leaves"]),
                                    min_split_gain="{:.3f}".format(
                                        packed_inputs["min_split_gain"]),
                                    min_child_weight="{:.3f}".format(
                                        packed_inputs["min_child_weight"]),
                                    subsample_for_bin=int(
                                        packed_inputs["subsample_for_bin"]),
                                    reg_alpha="{:.3f}".format(
                                        packed_inputs["reg_alpha"]),
                                    reg_lambda="{:.3f}".format(
                                        packed_inputs["reg_lambda"]),
                                    learning_rate="{:.3f}".format(packed_inputs["learning_rate"]))

        score = cross_val_score(self.model,
                                self.X_train.drop(['customer_ID'], axis=1),
                                np.ravel(self.Y_train),
                                scoring=self.amex_scorer,
                                cv=self.cfg["clf"]["search"]["cv"],
                                n_jobs=self.cfg["clf"]["init"]["n_jobs"]
                                ).mean()

        print(
            "Amex CV score = {:.3f}, params = {}".format(-score, packed_inputs))
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
