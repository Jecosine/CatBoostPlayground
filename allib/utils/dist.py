from typing import Callable

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from allib.typing import ArrayLike


# todo:
# 1. wasserstein_distance
# 2. jensenshannon

def get_dist_metric(name: str) -> Callable:
    def wrap_func(X: ArrayLike, Y: ArrayLike):
        return pairwise_distances(X, Y, metric=name)
    return wrap_func


def get_knn(xi: int, dist_mat: ArrayLike, k: int):
    return np.argsort(dist_mat[xi])[:k]
