from typing import Callable

import pandas as pd
from sklearn.utils import shuffle as sf
from sklearn.model_selection import train_test_split
from allib.models.al import ActiveLearningMetric
from allib.typing import ArrayLike

AVAIL_DATASET = {}


# todo:
# 1. statistics on a dataset
# 2.


class Dataset:
    _meta = {}

    def __init__(
            self,
            data: ArrayLike,
            label: ArrayLike,
            al_metric: ActiveLearningMetric,
            shuffle: bool,
            random_state: int,
            init_size: float | int,
            batch_size: int,
            # batch_size_updator: Callable,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        # force pandas dataframe
        self._data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self._label = label if isinstance(label, pd.DataFrame) else pd.DataFrame(label)
        self.random_state = random_state
        self.batch_size = batch_size
        # todo: support metrics switching?
        self.al_metric = al_metric
        if shuffle:
            self._data = sf(self._data, random_state=random_state)
            self._data.reset_index(drop=True)
        # split train/test set
        self._split_train_test()
        # if percentage, calculate the actual number of instances
        if init_size < 1:
            init_size = len(self.u_x) * init_size
        self._init_size = min(len(self.u_x), init_size)  # prevent overflow the length

        self.u_x: pd.DataFrame = pd.DataFrame()
        self.u_y: pd.DataFrame = pd.DataFrame()
        self.l_x: pd.DataFrame = pd.DataFrame()
        self.l_y: pd.DataFrame = pd.DataFrame()

    def _create_iter(self, train_x: ArrayLike, train_y: ArrayLike, batch_size: int):
        pass

    def _split_train_test(self):
        """ Split training and testing set. In active learning, the set `U` denotes the unlabeled dataset and `L`
        denotes the labeled set.
        """
        # todo: setting ratio
        self.u_x, self.u_y, self.test_x, self.test_x = train_test_split(
            self._data.copy(), self._label.copy(), random_state=self.random_state
        )

    def field_info(self):
        """Statistic on the field types. Save the result in `_meta`, called in init
        """
        # todo:
        #  1. record the categorical/numerical data amount
        #  2. count the possible output for categorical
        pass

    def __iter__(self):
        self._first_batch = True
        return self

    def __next__(self):
        """ next labeled set X, y

        Returns:
            pd.DataFrame: labeled set x
            pd.DataFrame: labeled set y
        """
        if self._first_batch:
            self.u_x, self.u_y, nx, ny = self.al_metric(
                self.u_x, self.u_y, initial_batch=True
            )
            self._first_batch = False
        else:
            self.u_x, self.u_y, nx, ny = self.al_metric(self.u_x, self.u_y)
        # update the `L` set
        self.l_x = pd.concat((self.l_x, nx))
        self.l_y = pd.concat((self.l_y, ny))
        return self.l_x, self.l_y
