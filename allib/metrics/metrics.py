from typing import List, Callable
from sklearn.metrics import get_scorer, get_scorer_names

from allib.models import BaseModel
from allib.typing import ArrayLike


def __name_metric(f: Callable, name: str):
    if f is None:
        return None

    def wrap_func(estimator: BaseModel, X: ArrayLike, y_true: ArrayLike):
        return f(estimator=estimator, X=X, y_true=y_true)

    wrap_func.__name__ = name
    return wrap_func


def get_metrics(names: List[str], ignore_error: bool = False):
    def suppress_error(name: str):
        try:
            func = get_scorer(name)
        except ValueError:
            return None
        else:
            return func

    getter = suppress_error if ignore_error else get_scorer
    return [
        func
        for name in names
        if (func := __name_metric(getter(name), name)) is not None
        # if (func := __name_metric(getter(name)._score_func)) is not None
    ]


def avail_metrics():
    return get_scorer_names()
