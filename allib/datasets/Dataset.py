import math
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sf

from allib.models.al import ActiveLearningMetric
from allib.typing import ArrayLike

AVAIL_DATASET = {}


# todo:
# 1. statistics on a dataset
# 2. dataset maker, using same data with multiple metrics
# 3. reset


class Dataset:
    """ Dataset class for active learning pipeline"""

    _meta = {}

    def __init__(
        self,
        data: ArrayLike,
        label: ArrayLike,
        al_metric: ActiveLearningMetric,
        shuffle: bool,
        init_size: float | int,
        batch_size: int,
        # batch_size_updator: Callable,
        random_state: int = 0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # force pandas dataframe
        self._data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self._label = label if isinstance(label, pd.DataFrame) else pd.DataFrame(label)
        self.random_state = random_state or 0
        self.batch_size = batch_size
        # todo: support metrics switching?
        self.al_metric = al_metric
        if shuffle:
            self._data = sf(self._data, random_state=random_state)
            self._data.reset_index(drop=True)
        self.u_x: pd.DataFrame = pd.DataFrame()
        self.u_y: pd.DataFrame = pd.DataFrame()
        self.l_x: pd.DataFrame = pd.DataFrame()
        self.l_y: pd.DataFrame = pd.DataFrame()
        self.reset()
        # if percentage, calculate the actual number of instances
        if init_size < 1:
            init_size = int(len(self.u_x) * init_size)
        self._init_size = min(len(self.u_x), init_size)  # prevent overflow the length

        self.__n_batches = math.ceil(
            (self.u_size - self._init_size) / self.batch_size
        ) + 1

        self.pipeline_params = {
            "model": None,
            "seeds": None,
        }

        self.model = None

    @property
    def u_size(self) -> int:
        """ Get the size of the set `U`

        Returns:
            int: size of the set `U`
        """
        return len(self.u_x)

    @property
    def l_size(self) -> int:
        """ Get the size of the set `L`

        Returns:
            int: size of the set `L`
        """
        return len(self.l_x)

    def _create_iter(self, train_x: ArrayLike, train_y: ArrayLike, batch_size: int):
        pass

    def _split_train_test(self):
        """ Split training and testing set. In active learning, the set `U` denotes the unlabeled dataset and `L`
        denotes the labeled set.
        """
        # todo: setting ratio
        self.u_x, self.test_x, self.u_y, self.test_y = train_test_split(
            self._data.copy(), self._label.copy(), random_state=self.random_state
        )

    def reset(self):
        # split train/test set
        self.u_x: pd.DataFrame = pd.DataFrame()
        self.u_y: pd.DataFrame = pd.DataFrame()
        self.l_x: pd.DataFrame = pd.DataFrame()
        self.l_y: pd.DataFrame = pd.DataFrame()
        self._split_train_test()

    def update_iteration(self, n_iter: Optional[int] = None):
        seeds = self.pipeline_params.get("seeds")
        # if not isinstance(seeds, list):
        #     raise RuntimeError("[Dataset] seeds should be list of int")
        self.random_state = seeds[n_iter]
        self.reset()

    def get_training_set(self) -> (pd.DataFrame, pd.DataFrame):
        return self.l_x, self.l_y

    def get_testing_set(self) -> (pd.DataFrame, pd.DataFrame):
        return self.test_x, self.test_y

    def bind_params(self, params: dict):
        self.pipeline_params = params
        self.model = self.pipeline_params.get("model")
        # self.random_state = self.pipeline_params.get("random_state")

    def field_info(self):
        """ Statistic on the field types. Save the result in `_meta`, called in init
        """
        # todo:
        #  1. record the categorical/numerical data amount
        #  2. count the possible output for categorical
        pass

    def __len__(self):
        return self.__n_batches

    def __iter__(self):
        self._first_batch = True
        return self

    def __next__(self):
        """ next labeled set X, y

        Returns:
            pd.DataFrame: labeled set x
            pd.DataFrame: labeled set y
        """
        if self.u_size == 0:
            raise StopIteration
        if self._first_batch:
            self.u_x, self.u_y, nx, ny = self.al_metric(
                self.u_x,
                self.u_y,
                initial_batch=True,
                model=self.model,
                random_state=self.random_state,
                batch_size=self.batch_size
            )
            self._first_batch = False
        else:
            self.u_x, self.u_y, nx, ny = self.al_metric(
                self.u_x, self.u_y, model=self.model,  random_state=self.random_state, batch_size=self.batch_size
            )
        # update snapshot
        self.pipeline_params["current_stat"]["snapshot"].append(self.al_metric.current_idx)
        # update the `L` set
        self.l_x = pd.concat((self.l_x, nx))
        self.l_y = pd.concat((self.l_y, ny))
        return self.l_x, self.l_y
