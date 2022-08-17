import os
import pickle
from interfaces import IModel
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer


class Hyperopt(IModel):
    def __init__(self, cfg, clf):
        super(Hyperopt, self).__init__(cfg)
        self.clf = clf
        self.metric = getattr(__import__("metric"),
                              cfg["modules"]["metric"])
        self.scorer = make_scorer(self.metric, needs_proba=True)

    def build(self):
        print("Building model...")

        if os.path.isfile(self.cfg["model"]["train"]["load_from"]):
            print("Loading existing model: %s" %
                  self.cfg["model"]["train"]["load_from"])
            self.model = pickle.load(
                open(self.cfg["model"]["train"]["load_from"], 'rb'))

        else:
            self.model = self.clf(**self.cfg["model"]["clf"]["init"])

    def train(self, X_train, Y_train):
        assert (isinstance(self.model, self.clf)
                ), "Model is not yet built!"

        if not self.cfg["model"]["train"]["train"]:
            print("You chose to skip training...")
            return self.model, {}

        self.X_train = X_train
        self.Y_train = Y_train

        # Load params that are indicated to be searched in the cfg
        search_cfg = self.cfg["model"]["clf"]["searching_space"]
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
                    max_evals=self.cfg["model"]["train"]["itr"]
                    )

        # exclude params that are not in the searching space
        params_not_to_search = [
            param for param in self.cfg["model"]["clf"]["init"].keys() if param not in params_to_search]
        excluded = {param: self.cfg["model"]["clf"]["init"][param]
                    for param in params_not_to_search}

        # cast to int/float based on the params' types at init stage
        best_params = {param: int(best[param]) if isinstance(
            self.cfg["model"]["clf"]["init"][param], int) else float(best[param]) for param in best.keys()}

        final_model = self.clf(
            **excluded,
            **best_params
        )

        final_model.fit(self.X_train, self.Y_train)
        self.model = final_model

        return self.model, best_params

    def objective(self, packed_inputs):
        score = cross_val_score(self.model,
                                self.X_train,
                                self.Y_train,
                                scoring=self.scorer,
                                cv=self.cfg["model"]["clf"]["cross_validation"],
                                n_jobs=self.cfg["model"]["clf"]["init"]["n_jobs"]
                                ).mean()

        print("CV score = {:.6f}, params = {}".format(-score, packed_inputs))
        return score

    def eval(self, X_test, Y_test):
        assert (isinstance(self.model, self.clf)
                ), "Model is not yet built!"

        print("Evaluating model...")
        Y_pred = self.model.predict_proba(X_test)[:, 1]

        score = -self.metric(Y_test, Y_pred)

        print("Score: %.6f" % (score))

        return score
