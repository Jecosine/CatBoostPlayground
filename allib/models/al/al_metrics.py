from typing import Optional, Callable, Any

import numpy as np
import pandas as pd
from allib.typing import ArrayLike
from ..core import BaseModel


class ActiveLearningMetric:
    """ instance selection metrics for active learning """

    def __init__(
            self,
            model: BaseModel,
            batch_size: Optional[int],
            init_size: Optional[float | int],
    ):
        """ init the metric and return callable function

        Args:
            init_size:  percentage | n samples. Default equal to batch size if it is not specified
            model: the model you want to evaluate
            batch_size: optional parameter. use batching if set otherwise sequential
        """
        self.batch = batch_size is not None
        self.batch_size = batch_size
        self.model = model
        # assert hasattr(self.model, "fit")
        # if percentage, calculate the actual number of instances
        self.init_size = init_size or batch_size

    def sample_initial(self, train_x: ArrayLike, train_y: ArrayLike, random_state: int):
        pass

    def sample(
            self,
            train_x: ArrayLike,
            train_y: ArrayLike,
            random_state: int,
            initial_batch: bool = False,
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
        N = self.batch_size
        if initial_batch:
            # if percentage, calculate the actual number of instances
            if self.init_size < 1:
                N = len(train_x) * self.init_size
            N = min(len(train_x), N)  # prevent overflow the length
        batch_x = train_x.sample(n=N, random_state=random_state)
        batch_y = train_y.loc[batch_x.index]
        return (
            train_x.drop(batch_x.index),
            train_y.drop(batch_y.index),
            batch_x,
            batch_y,
        )

    def __call__(
            self,
            train_x: ArrayLike,
            train_y: ArrayLike,
            random_state: int,
            initial_batch: bool = False,
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
        return self.sample(train_x, train_y, **kwargs)


# todo:
# 1. kmeans++
# 2. k-PP
# 3.
class RandomMetric(ActiveLearningMetric):
    """ New instances added by random batch selection """

    pass


class UncertainMetric(ActiveLearningMetric):
    def sample_initial(self, train_x: ArrayLike, train_y: ArrayLike, random_state: int):
        N = self.init_size
        # if percentage, calculate the actual number of instances
        if self.init_size < 1:
            N = len(train_x) * self.init_size
        N = min(len(train_x), N)  # prevent overflow the length
        batch_x = train_x.sample(n=N, random_state=random_state)
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
            random_state: int,
            initial_batch: bool = False,
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
        if initial_batch:
            return self.sample_initial(train_x, train_y, random_state)

        if len(train_x) > self.batch_size:
            idx = (
                self.model.predict_proba(train_x)
                .max(axis=1)
                .argpartition(self.batch_size)[: self.batch_size]
            )

            return (
                train_x.drop(train_x.index[idx]),
                train_y.drop(train_y.index[idx]),
                train_x.loc[train_x.index[idx]],
                train_y[train_y.index[idx]],
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
