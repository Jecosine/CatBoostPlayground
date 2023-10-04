from typing import List

import numpy as np
import os
from allib.typing import ArrayLike


def arg_bottomk(arr: ArrayLike, k: int) -> ArrayLike:
    return np.argpartition(arr, k)[: k]


def arg_topk(arr: ArrayLike, k: int) -> ArrayLike:
    return np.argpartition(arr, -k)[-k:]


def make_seeds(n: int = 10) -> List[int]:
    return [int.from_bytes(os.urandom(4), "big") for _ in range(n)]
