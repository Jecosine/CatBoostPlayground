from typing import Optional

from catboost import CatBoostClassifier
from sklearn import svm
from sklearn.base import clone as sk_clone
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from copy import deepcopy
from .al import ActiveLearningPipeline


class CatboostPL(ActiveLearningPipeline):
    _default_model_params = {
        "iterations": 5,
        "learning_rate": 0.1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_params = {"cat_features": kwargs.get("cat_idx")}
        self.model_maker = self.model_maker or self.__model_maker

    def run(self, n_iter: Optional[int] = None, extra_params=None):
        if extra_params is None:
            extra_params = {}
        # prog = tqdm(self.dataset, total=len(self.dataset))
        # prog.set_description(f"The {n_iter+1}th run:")
        self.model = self.model_maker()
        counter = 0
        for tx, ty in self.dataset:
            if counter >= self.early_stop:
                break
            self.model.fit(tx, ty, verbose=False, **extra_params)
            self.current_stat["model_snapshots"].append(self.model.copy())
            self.apply_eval_metrics()

    def __model_maker(self, params: dict = None):
        if params is None:
            params = {}
        params = self._default_model_params | params
        return CatBoostClassifier(**params)


class SVCPL(ActiveLearningPipeline):
    _default_model_params = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_maker = self.model_maker or self.__model_maker

    def run(self, n_iter: Optional[int] = None, extra_params=None):
        extra_params = extra_params or {}
        self.model = self.model_maker()
        counter = 0
        for tx, ty in self.dataset:
            if counter >= self.early_stop:
                break
            self.model.fit(tx, ty, **extra_params)
            self.current_stat["model_snapshots"].append(deepcopy(self.model))
            self.apply_eval_metrics()

    def __model_maker(self, params: dict = None):
        if params is None:
            params = {}
        params = self._default_model_params | params
        return svm.SVC(**params)


class LogisticRegressionPL(ActiveLearningPipeline):
    _default_model_params = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_maker = self.model_maker or self.__model_maker

    def run(self, n_iter: Optional[int] = None, extra_params=None):
        extra_params = extra_params or {}
        self.model = self.model_maker()
        counter = 0
        for tx, ty in self.dataset:
            if counter >= self.early_stop:
                break
            self.model = self.model.fit(tx, ty, **extra_params)
            self.current_stat["model_snapshots"].append(deepcopy(self.model))
            self.apply_eval_metrics()

    def __model_maker(self, params: dict = None):
        if params is None:
            params = {}
        params = self._default_model_params | params
        return LogisticRegression(**params)


class MLPPL(ActiveLearningPipeline):
    _default_model_params = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_maker = self.model_maker or self.__model_maker

    def run(self, n_iter: Optional[int] = None, extra_params=None):
        extra_params = extra_params or {}
        self.model = self.model_maker()
        counter = 0
        for tx, ty in self.dataset:
            if counter >= self.early_stop:
                break
            self.model = self.model.fit(tx, ty, **extra_params)
            self.current_stat["model_snapshots"].append(deepcopy(self.model))
            self.apply_eval_metrics()

    def __model_maker(self, params: dict = None):
        if params is None:
            params = {}
        params = self._default_model_params | params
        return MLPClassifier(**params)


AVAIL_MODEL = {
    "catboost": CatboostPL,
    "svm": SVCPL,
    "logistic": LogisticRegressionPL,
    "mlp": MLPPL,
}


def get_pipeline(name: str) -> ActiveLearningPipeline:
    ppl = AVAIL_MODEL.get(name, None)
    if ppl is None:
        raise RuntimeError(f"Model {name} is not implemented")
    return ppl
