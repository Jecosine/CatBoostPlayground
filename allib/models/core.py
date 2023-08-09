import numpy as np
import pandas as pd

from allib.typing import ArrayLike


class BaseModel:
    def __init__(self):
        pass

    def fit(self):
        pass

    def fit_transform(self):
        pass

    def score(self):
        pass

    def predict(self, data: ArrayLike) -> ArrayLike:
        pass

    def predict_proba(self, data: ArrayLike) -> ArrayLike:
        pass

    def __get_metric(self):
        pass
