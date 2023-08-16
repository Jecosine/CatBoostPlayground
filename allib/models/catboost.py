from catboost import CatBoostClassifier as CBC

from .core import BaseModel
from ..datasets import Dataset


class CatBoostClassifier(BaseModel):
    """ Catboost Classifier """

    def __init__(self, *args, **kwargs):
        self._model: CBC = CBC(*args, **kwargs)
        super().__init__()

    def fit(self, *args, **kwargs):
        self._model.fit(*args, **kwargs)

    def al_fit(self, training_set: Dataset, batch: bool = False):
        pass
