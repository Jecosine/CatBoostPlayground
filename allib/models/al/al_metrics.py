from typing import Optional, Callable, Any

import numpy as np
import pandas as pd
from allib.typing import ArrayLike
from ..core import BaseModel
from allib.datasets import Dataset
from abc import ABC, abstractmethod


class ActiveLearningMetric(ABC):
    """ instance selection metrics for active learning """

    def __init__(
        self,
        # model: BaseModel,
        batch_size: Optional[int] = 10,
        init_size: Optional[float | int] = None,
        random_state: Optional[int] = 0,
    ):
        """ init the metric and return callable function

        Args:
            init_size:  percentage | n samples. Default equal to batch size if it is not specified
            batch_size: optional parameter. use batching if set otherwise sequential
            random_state: random state
        """
        self.batch = batch_size is not None
        self.batch_size = batch_size
        # self.model = model
        # assert hasattr(self.model, "fit")
        # if percentage, calculate the actual number of instances
        self.init_size = init_size or batch_size
        self.random_state = random_state

    @abstractmethod
    def sample_initial(
        self, train_x: ArrayLike, train_y: ArrayLike, random_state: Optional[int] = None
    ):
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        train_x: ArrayLike,
        train_y: ArrayLike,
        random_state: Optional[int] = None,
        initial_batch: bool = False,
        *args,
        **kwargs,
    ) -> (ArrayLike, ArrayLike, ArrayLike, ArrayLike):
        """ sample by random sampling

        Args:
            initial_batch: true if this batch is the initial set
            train_x: training set X
            train_y: training set Y
            random_state: random seed

        Returns:
            ArrayLike: truncated training x
            ArrayLike: truncated training y
            ArrayLike: newly sampled batch x
            ArrayLike: newly sampled batch y

        """
        raise NotImplementedError

    def __call__(
        self,
        train_x: ArrayLike,
        train_y: ArrayLike,
        random_state: Optional[int] = None,
        initial_batch: bool = False,
        *args,
        **kwargs,
    ) -> (ArrayLike, ArrayLike, ArrayLike, ArrayLike):
        """ Return a sample function that returns

        Args:
            **kwargs: config params

        Returns:
            ArrayLike: truncated training x
            ArrayLike: truncated training y
            ArrayLike: newly sampled batch x
            ArrayLike: newly sampled batch y
        """
        return self.sample(
            train_x,
            train_y,
            random_state=random_state,
            initial_batch=initial_batch,
            *args,
            **kwargs,
        )


# todo:
# 1. kmeans++
# 2. k-PP
# 3.
class RandomMetric(ActiveLearningMetric):
    """ New instances added by random batch selection """

    def sample_initial(
        self, train_x: ArrayLike, train_y: ArrayLike, random_state: Optional[int] = None
    ):
        pass

    def sample(
        self,
        train_x: ArrayLike,
        train_y: ArrayLike,
        random_state: Optional[int] = None,
        initial_batch: bool = False,
        *args,
        **kwargs,
    ) -> (ArrayLike, ArrayLike, ArrayLike, ArrayLike):
        N = self.batch_size
        if initial_batch:
            # if percentage, calculate the actual number of instances
            if self.init_size < 1:
                N = int(len(train_x) * self.init_size)
        N = min(len(train_x), N)  # prevent overflow the length
        batch_x = train_x.sample(n=N, random_state=random_state)
        batch_y = train_y.loc[batch_x.index]
        return (
            train_x.drop(batch_x.index),
            train_y.drop(batch_y.index),
            batch_x,
            batch_y,
        )


class UncertainMetric(ActiveLearningMetric):
    def sample_initial(
        self, train_x: ArrayLike, train_y: ArrayLike, random_state: Optional[int] = None
    ):
        N = self.init_size
        # if percentage, calculate the actual number of instances
        if self.init_size < 1:
            N = int(len(train_x) * self.init_size)
        N = min(len(train_x), N)  # prevent overflow the length
        batch_x = train_x.sample(n=N, random_state=random_state or self.random_state)
        batch_y = train_y.loc[batch_x.index]
        return (
            train_x.drop(batch_x.index),
            train_y.drop(batch_y.index),
            batch_x,
            batch_y,
        )

    def sample(
        self,
        train_x: ArrayLike,
        train_y: ArrayLike,
        random_state: Optional[int] = None,
        initial_batch: bool = False,
        *args,
        **kwargs,
    ):
        """ Sample based on the uncertainty of unlabeled instances

        Args:
            initial_batch: if is initial batch
            train_x: training set X
            train_y: training set Y
            random_state: random seed

        Returns:
            ArrayLike: truncated training x
            ArrayLike: truncated training y
            ArrayLike: newly sampled batch x
            ArrayLike: newly sampled batch y

        """
        model = kwargs.get("model", None)
        if model is None:
            raise RuntimeError(f"Uncertainty metric requires a `model` parameter")

        if initial_batch:
            return self.sample_initial(train_x, train_y, random_state)

        if len(train_x) > self.batch_size:
            idx = (
                model.predict_proba(train_x)
                .max(axis=1)
                .argpartition(self.batch_size)[: self.batch_size]
            )

            return (
                train_x.drop(train_x.index[idx]),
                train_y.drop(train_y.index[idx]),
                train_x.loc[train_x.index[idx]],
                train_y.loc[train_y.index[idx]],
            )
        else:
            return (
                train_x.drop(train_x.index),
                train_y.drop(train_y.index),
                train_x.copy(),
                train_y.copy(),
            )


__ALL_METRICS = {"random": RandomMetric, "uncertain": UncertainMetric}


def get_al_metric(name: str, params: dict):
    if name in __ALL_METRICS:
        return __ALL_METRICS[name](**params)
    else:
        raise RuntimeError(
            f"Metric {name} not exists. For custom metric, please override the `ActiveLearningMetric` class"
        )
