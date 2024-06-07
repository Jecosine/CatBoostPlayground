import math
from typing import Optional, Type, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sf
from .preprocess import build_preprocess_ppl
from allib.models.al import ActiveLearningStrategy
from allib.typing import ArrayLike
from copy import deepcopy
from .tools import get_cat_idx, remove_inf


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
        al_strategy: ActiveLearningStrategy | None,
        shuffle: bool,
        init_size: float | int,
        batch_size: int,
        # batch_size_updator: Callable,
        random_state: int = 0,
        *args,
        **kwargs
    ):
        # force pandas dataframe
        self._data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self._label = label if isinstance(label, pd.DataFrame) else pd.DataFrame(label)
        self.random_state = random_state or 0
        self.batch_size = batch_size
        # todo: support strategy switching?
        self.al_metric = al_strategy
        if shuffle:
            self._data = sf(self._data, random_state=random_state)
            self._data.reset_index(drop=True)
        self.u_x: pd.DataFrame = pd.DataFrame()
        self.u_y: pd.DataFrame = pd.DataFrame()
        self.l_x: pd.DataFrame = pd.DataFrame()
        self.l_y: pd.DataFrame = pd.DataFrame()
        
        self.info = {"cat_idx": []}
        self.info["cat_idx"] = get_cat_idx(self._data)
        
        self._split_train_test()
        # if percentage, calculate the actual number of instances
        if init_size < 1:
            init_size = int(len(self.u_x) * init_size)
        self._init_size = min(len(self.u_x), init_size)  # prevent overflow the length

        self.__n_batches = (
            math.ceil((self.u_size - self._init_size) / self.batch_size) + 1
        )

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
        self.info["cat_idx"] = get_cat_idx(self._data)
        self._split_train_test()
        # update batch size
        self.__n_batches = (
            math.ceil((self.u_size - self._init_size) / self.batch_size) + 1
        )

    def update_iteration(self, n_iter: Optional[int] = None):
        # todo assert seeds
        seeds: list = self.pipeline_params.get("seeds", [])
        if len(seeds) == 0:
            raise RuntimeError("[DATASET]: Seeds is required by Dataset method `update_iteration`")
        # if not isinstance(seeds, list):
        #     raise RuntimeError("[Dataset] seeds should be list of int")
        self.random_state = seeds[n_iter]
        self.reset()

    def with_strategy(
        self, strategy: Type[ActiveLearningStrategy], extra_params: dict = None
    ):
        if extra_params is None:
            extra_params = {}
        extra_params = extra_params | {"info": self.info}

        dataset = Dataset(
            data=self._data.copy(),
            label=self._label.copy(),
            al_strategy=strategy(
                **(
                    {"init_size": self._init_size, "batch_size": self.batch_size}
                    | extra_params
                )
            ),
            **{
                "shuffle": False,
                "init_size": self._init_size,
                "batch_size": self.batch_size,
            }
        )
        dataset.info = self.info
        return dataset

    def with_preprocess(self, steps: List[str], params_list: List[dict], in_place: bool = True) -> Optional["Dataset"]:
        """ preprocess dataset

        Args:
            steps: specify preprocess
            params_list: params for steps
            in_place: if True just modify the properties in place, otherwise return new dataset instance.

        Returns:
            Optional[Dataset]: preprocessed dataset
        """
        pppl = build_preprocess_ppl(steps, params_list)
        data, label = pppl(self._data, self._label)
        # temp_data = remove_inf(temp_data)
        if in_place:
            self._data = data
            self._label = label
            self.reset()
        else:
            clone = deepcopy(self)
            clone._data = data
            clone._label = label
            clone.reset()
            return clone

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
                batch_size=self.batch_size,
                L=self.l_x,
                u_size=self.u_size,
                l_x=self.l_x,
            )
            self._first_batch = False
        else:
            # try to get weight information
            fi = self.pipeline_params["current_stat"].get("fi", None)
            n_w, c_w = 0.0, 0.0
            if fi is not None:
                fi = fi[0]
                fi /= sum(fi)
                # weight of numerical
                for i in self.info["cat_idx"]:
                    c_w += fi[i]
                n_w = 1.0 - c_w
            self.u_x, self.u_y, nx, ny = self.al_metric(
                self.u_x,
                self.u_y,
                model=self.model,
                random_state=self.random_state,
                batch_size=self.batch_size,
                L=self.l_x,
                u_size=self.u_size,
                l_x=self.l_x,
                n_w=n_w,
                c_w=c_w,
            )
        # update snapshot
        self.pipeline_params["current_stat"]["snapshot"].append(
            self.al_metric.current_idx
        )
        # update the `L` set
        self.l_x = pd.concat((self.l_x, nx))
        self.l_y = pd.concat((self.l_y, ny))
        return self.l_x, np.ravel(self.l_y)
