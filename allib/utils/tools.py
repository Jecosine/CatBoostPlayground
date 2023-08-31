import numpy as np

from allib.typing import ArrayLike


def arg_bottomk(arr: ArrayLike, k: int) -> ArrayLike:
    return np.argpartition(arr, k)[: k]


def arg_topk(arr: ArrayLike, k: int) -> ArrayLike:
    return np.argpartition(arr, -k)[-k:]

