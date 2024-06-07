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


def get_metrics(names: List[str], ignore_error: bool = False, param_list: List[dict] = None):
    if param_list is None:
        param_list = [{} for _ in range(len(names))]
    def suppress_error(name: str, **params):
        try:
            func = get_scorer(name, **params)
        except ValueError:
            return None
        else:
            return func

    getter = suppress_error if ignore_error else get_scorer
    return [
        func
        for name, params in zip(names, param_list)
        if (func := __name_metric(getter(name, **params), name)) is not None
        # if (func := __name_metric(getter(name)._score_func)) is not None
    ]


def avail_metrics():
    return get_scorer_names()
