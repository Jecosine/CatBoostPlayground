from abc import ABC, abstractmethod
from typing import Optional, Callable

import numpy as np
import pandas as pd
from scipy.stats import entropy, norm
from sklearn.cluster import KMeans
from allib.typing import ArrayLike
from allib.utils import arg_bottomk, arg_topk, get_dist_metric


class ActiveLearningStrategy(ABC):
    """ instance selection metrics for active learning """

    def __init__(
        self,
        # model: BaseModel,
        batch_size: Optional[int] = 10,
        init_size: Optional[float | int] = None,
        random_state: Optional[int] = 0,
        *args,
        **kwargs,
    ):
        """ init the metric and return callable function

        Args:
            init_size:  percentage | n samples. Default equal to batch size if it is not specified
            batch_size: optional parameter. use batching if set otherwise sequential
            random_state: random state
        """
        self.batch = batch_size
        self.batch_size = batch_size
        # self.model = model
        # assert hasattr(self.model, "fit")
        # if percentage, calculate the actual number of instances
        self.init_size = init_size or batch_size
        self.random_state = random_state
        self.current_idx = None

    def sample_initial(
        self, train_x: ArrayLike, train_y: ArrayLike, random_state: Optional[int] = None
    ):
        N = self.init_size
        # if percentage, calculate the actual number of instances
        if self.init_size < 1:
            N = int(len(train_x) * self.init_size)
        N = min(len(train_x), N)  # prevent overflow the length
        batch_x = train_x.sample(
            n=N,
            random_state=random_state
            if random_state is not None
            else self.random_state,
        )
        batch_y = train_y.loc[batch_x.index]
        self.current_idx = batch_x.index.copy()
        return (
            train_x.drop(batch_x.index),
            train_y.drop(batch_y.index),
            batch_x,
            batch_y,
        )

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
class RandomStrategy(ActiveLearningStrategy):
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
            else:
                N = self.init_size
        N = min(len(train_x), N)  # prevent overflow the length
        batch_x = train_x.sample(n=N, random_state=random_state)
        batch_y = train_y.loc[batch_x.index]
        self.current_idx = batch_x.index.copy()
        return (
            train_x.drop(batch_x.index),
            train_y.drop(batch_y.index),
            batch_x,
            batch_y,
        )


class UncertainStrategy(ActiveLearningStrategy):
    def __init__(
        self, name: str = "uncertainty", *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.name = name
        self.func_strategy = {
            "uncertainty": self.__func_uncertainty,
            "margin": self.__func_margin,
            "entropy": self.__func_entropy,
        }[self.name]

    def __func_uncertainty(self, pred: ArrayLike):
        return arg_bottomk(pred.max(axis=1), self.batch_size)

    def __func_margin(self, pred: ArrayLike):
        margin = np.partition(-pred, 1, axis=1)
        margin = -margin[:, 0] + margin[:, 1]
        return arg_bottomk(margin, self.batch_size)

    def __func_entropy(self, pred: ArrayLike):
        e = entropy(pred.T).squeeze()
        return arg_topk(e, self.batch_size)

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
            pred = model.predict_proba(train_x)
            idx = self.func_strategy(pred)
            self.current_idx = train_x.index[idx].copy()
            return (
                train_x.drop(train_x.index[idx]),
                train_y.drop(train_y.index[idx]),
                train_x.loc[train_x.index[idx]],
                train_y.loc[train_y.index[idx]],
            )
        else:
            self.current_idx = train_x.index.copy()
            return (
                train_x.drop(train_x.index),
                train_y.drop(train_y.index),
                train_x.copy(),
                train_y.copy(),
            )


class DisagreementStrategy(ActiveLearningStrategy):
    def __init__(
        self,
        make_model: Callable,
        n_committees: int = 5,
        name: str = "vote",
        info: dict = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.name = name
        if make_model is None:
            raise RuntimeError("Disagreement strategy need model maker to init models")
        if n_committees < 2:
            raise RuntimeError("Disagreement strategy need more than 1 committees")
        self.info = info or {}
        extra_params = (
            {"cat_features": self.info["cat_idx"]} if self.info.get("cat_idx") else {}
        )
        # extra_params = {}
        self.models = [make_model(extra_params) for _ in range(n_committees - 1)]
        self.n_committees = n_committees
        self.func_strategy = {
            "vote": self.__func_vote,
            "consensus": self.__func_consensus,
            "max_disagreement": self.__func_max_disagreement,
        }[self.name]
        self.l_x = pd.DataFrame()
        self.l_y = pd.DataFrame()

    def __func_vote(self, pred: ArrayLike):
        res = pred.argmax(axis=-1).T
        n_votes = res.shape[-1]
        votes = np.zeros((res.shape[0], pred.shape[-1]))
        for i in range(votes.shape[0]):
            unique, count = np.unique(res[i], return_counts=True)
            votes[i, unique] = count / n_votes
        v_e = entropy(votes.T).squeeze()
        return arg_topk(v_e, self.batch_size)

    def __func_consensus(self, pred: ArrayLike):
        res = pred.mean(axis=0)
        c_e = entropy(res.T).squeeze()
        return arg_topk(c_e, self.batch_size)

    def __func_max_disagreement(self, pred: ArrayLike):
        # get consensus
        res = pred.mean(axis=0)
        c_e = entropy(res.T).squeeze()
        n_samples, n_votes = res.shape[0], pred.shape[0]
        kl = np.zeros((n_samples, n_votes))
        for i in range(n_samples):
            for j in range(n_votes):
                kl[i, j] = entropy(pred[j, i], qk=res[i])
        kl = kl.max(axis=1)
        return arg_topk(kl, self.batch_size)

    def sample(
        self,
        train_x: ArrayLike,
        train_y: ArrayLike,
        random_state: Optional[int] = None,
        initial_batch: bool = False,
        *args,
        **kwargs,
    ):
        model = kwargs.get("model", None)
        if model is None:
            raise RuntimeError(f"Disagreement metric requires a `model` parameter")
        if initial_batch:
            u_x, u_y, self.l_x, self.l_y = self.sample_initial(
                train_x, train_y, random_state
            )
            return u_x, u_y, self.l_x.copy(), self.l_y.copy()

        for m in self.models:
            m.fit(self.l_x, np.ravel(self.l_y))
        models = [model, *self.models]
        if len(train_x) > self.batch_size:
            pred = np.array([m.predict_proba(train_x) for m in models])
            idx = self.func_strategy(pred)
            self.current_idx = train_x.index[idx].copy()
            l_x = train_x.loc[train_x.index[idx]]
            l_y = train_y.loc[train_x.index[idx]]
            self.l_x = pd.concat((self.l_x, l_x))
            self.l_y = pd.concat((self.l_y, l_y))
            return (
                train_x.drop(train_x.index[idx]),
                train_y.drop(train_y.index[idx]),
                train_x.loc[train_x.index[idx]],
                train_y.loc[train_y.index[idx]],
            )
        else:
            self.current_idx = train_x.index.copy()
            return (
                train_x.drop(train_x.index),
                train_y.drop(train_y.index),
                train_x.copy(),
                train_y.copy(),
            )


class RankedBatchModeStrategy(ActiveLearningStrategy):
    def __init__(self, dist_metric: str = "cosine", *args, **kwargs):
        super().__init__(*args, **kwargs)
        # todo: distance functions
        self.dist_func = get_dist_metric(dist_metric)
        self.sim = {"cosine": self.__cosine_sim, "euclidean": self.__euclidean_sim,}[
            dist_metric
        ]

    def __cosine_sim(self, X: ArrayLike, Y: ArrayLike):
        return self.dist_func(X, Y).max(axis=1)

    def __euclidean_sim(self, X: ArrayLike, Y: ArrayLike):
        return 1.0 / (1.0 + self.dist_func(X, Y).min(axis=1))

    def sample(
        self,
        train_x: ArrayLike,
        train_y: ArrayLike,
        random_state: Optional[int] = None,
        initial_batch: bool = False,
        *args,
        **kwargs,
    ) -> (ArrayLike, ArrayLike, ArrayLike, ArrayLike):
        model = kwargs.get("model", None)
        alpha: float = kwargs.get("alpha", 0.4)
        L: ArrayLike = kwargs.get("L", None)
        if model is None:
            raise RuntimeError(f"Ranked Batch Mode metric requires a `model` parameter")
        if alpha is None:
            raise RuntimeError(f"Ranked Batch Mode metric requires a `alpha` parameter")
        if L is None:
            raise RuntimeError(f"Ranked Batch Mode metric requires a `L` parameter")

        if initial_batch:
            return self.sample_initial(train_x, train_y, random_state)
        if len(train_x) > self.batch_size:
            uncertainty = 1 - model.predict_proba(train_x).max(axis=1)
            score = alpha * (1 - self.sim(train_x, L)) + (1 - alpha) * uncertainty
            idx = arg_topk(score, self.batch_size)
            self.current_idx = train_x.index[idx].copy()
            return (
                train_x.drop(train_x.index[idx]),
                train_y.drop(train_y.index[idx]),
                train_x.loc[train_x.index[idx]],
                train_y.loc[train_y.index[idx]],
            )
        else:
            self.current_idx = train_x.index.copy()
            return (
                train_x.drop(train_x.index),
                train_y.drop(train_y.index),
                train_x.copy(),
                train_y.copy(),
            )


class InformationDensityStrategy(ActiveLearningStrategy):
    def __init__(self, similarity_metric: str = "cosine", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sim = get_dist_metric(name=similarity_metric)

    def sample(
        self,
        train_x: ArrayLike,
        train_y: ArrayLike,
        random_state: Optional[int] = None,
        initial_batch: bool = False,
        *args,
        **kwargs,
    ) -> (ArrayLike, ArrayLike, ArrayLike, ArrayLike):
        u_size: int = kwargs.get("u_size", None)
        if u_size is None:
            raise RuntimeError(
                f"Information density metric requires a `u_size` parameter"
            )

        if initial_batch:
            return self.sample_initial(train_x, train_y, random_state)
        if len(train_x) > self.batch_size:
            info = self.sim(train_x, train_x).mean(axis=1)
            # todo
            idx = arg_topk(info, self.batch_size)
            self.current_idx = train_x.index[idx].copy()
            return (
                train_x.drop(train_x.index[idx]),
                train_y.drop(train_y.index[idx]),
                train_x.loc[train_x.index[idx]],
                train_y.loc[train_y.index[idx]],
            )
        else:
            self.current_idx = train_x.index.copy()
            return (
                train_x.drop(train_x.index),
                train_y.drop(train_y.index),
                train_x.copy(),
                train_y.copy(),
            )


class AcquisitionStrategy(ActiveLearningStrategy):
    def __init__(self, acq_name: str = "PI", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acquisition_func = {
            "PI": self.__pi_func,
            "EI": self.__ei_func,
            "ucb": self.__ucb_func,
        }[acq_name]

    def __pi_func(self, mu, peak, sigma, xi: float = 0, *args, **kwargs):
        return norm.cdf((mu - peak - xi) / sigma)

    def __ei_func(self, mu, peak, sigma, xi: float = 0, *args, **kwargs):
        return (mu - peak - xi) * self.__pi_func(mu, peak, sigma, xi) + norm.pdf(
            (mu - peak - xi) / sigma
        ) * sigma

    def __ucb_func(self, mu, sigma, beta, *args, **kwargs):
        return mu + beta * sigma

    def sample(
        self,
        train_x: ArrayLike,
        train_y: ArrayLike,
        random_state: Optional[int] = None,
        initial_batch: bool = False,
        *args,
        **kwargs,
    ) -> (ArrayLike, ArrayLike, ArrayLike, ArrayLike):
        # the model should be bayes optimizer returning mu and sigma
        model = kwargs.get("model", None)
        model_params = kwargs.get("model_params", {})
        # beta =
        if model is None:
            raise RuntimeError(f"Ranked Batch Mode metric requires a `model` parameter")
        mu, sigma = model.predict(train_x, **model_params)
        # acq = self.acquisition_func(mu=mu, sigma=sigma, )
        pass


class GSxStrategy(ActiveLearningStrategy):
    def __init__(self, distance_metric: str = "cosine", *args, **kwargs):
        super().__init__(*args, **kwargs)
        # todo: add params to get_dist_metric
        self.__dist = get_dist_metric(name=distance_metric)
        self.__cache_d = None

    def dist(self, u_x: pd.DataFrame, l_x: pd.DataFrame):
        """ get distance matrix
        Args:
            u_x: unlabeled data (n1, m)
            l_x: labeled data   (n2, m)

        Returns:
            np.ndarray: shape (n1, n1 + n2)
            pd.Index: indices of unlabeled data
            pd.Index: indices of labeled data
        """
        ux_idx = list(u_x.index)
        lx_idx = list(l_x.index)
        # todo: get cache
        if self.__getattribute__("dist_cache_path") is not None:
            cache_path = self.__getattribute__("dist_cache_path")
            # debug
            # print(f"loading cache {cache_path}...")
            if self.__cache_d is None:
                self.__cache_d = np.load(cache_path)

            return self.__cache_d[np.array(ux_idx)[:, np.newaxis], np.array(lx_idx + ux_idx)[np.newaxis, :]], ux_idx, lx_idx

        # ux_idx = {i: j for i, j in enumerate(ux_idx)}
        # lx_idx = {i: j for i, j in enumerate(lx_idx)}
        _u_x = u_x.to_numpy()
        _l_x = np.vstack((l_x.to_numpy(), _u_x))
        return self.__dist(u_x, _l_x), ux_idx, lx_idx

    def __get_batch(self, l_x: pd.DataFrame, u_x: pd.DataFrame, k: int):
        d, ux_idx, lx_idx = self.dist(u_x, l_x)
        batch = []
        r = np.array(list(range(len(ux_idx))))
        c = list(range(len(lx_idx)))
        c_max = len(c)
        for i in range(k):
            selected = d[r[:, np.newaxis], np.array(c)[np.newaxis, :]].min(axis=1).argmax()
            batch.append(selected)
            d[selected, :] = -np.inf
            c.append(selected + c_max)
        return batch
        # # centroid
        # c = data.mean(axis=0)
        # # center
        # c_idx = self.sim([c], data).argmin()
        # # k-1 batch
        # d = self.sim(data, data)
        # d += np.diag(np.inf * np.ones(d.shape[0]))
        # batch = [c_idx]
        # for i in range(k - 1):
        #     idx = d.min(axis=1).argmax()
        #     batch.append(idx)
        #     d[:, idx] = np.inf
        #     d[idx, :] = -1
        # return batch

    def sample(
        self,
        train_x: ArrayLike,
        train_y: ArrayLike,
        random_state: Optional[int] = None,
        initial_batch: bool = False,
        *args,
        **kwargs,
    ) -> (ArrayLike, ArrayLike, ArrayLike, ArrayLike):
        # todo: add utils for parameter check
        u_size: int = kwargs.get("u_size", None)
        l_x: ArrayLike = kwargs.get("l_x", None)
        if u_size is None:
            raise RuntimeError(f"GSx metric requires a `u_size` parameter")
        if l_x is None:
            raise RuntimeError(f"GSx metric requires a `l_x` parameter")
        if initial_batch:
            return self.sample_initial(train_x, train_y, random_state)
        if len(train_x) > self.batch_size:
            idx = self.__get_batch(l_x, train_x, self.batch_size)
            self.current_idx = train_x.index[idx].copy()
            return (
                train_x.drop(train_x.index[idx]),
                train_y.drop(train_y.index[idx]),
                train_x.loc[train_x.index[idx]],
                train_y.loc[train_y.index[idx]],
            )
        else:
            self.current_idx = train_x.index.copy()
            return (
                train_x.drop(train_x.index),
                train_y.drop(train_y.index),
                train_x.copy(),
                train_y.copy(),
            )


class TypiClust(ActiveLearningStrategy):
    def __init__(self, k: int = 5, distance_metric: str = "cosine", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.__dist = get_dist_metric(name=distance_metric)
        self.__cache_d = None

    def dist(self, u_x: pd.DataFrame):
        """ get distance matrix
        Args:
            u_x: unlabeled data (n1, m)
            l_x: labeled data   (n2, m)

        Returns:
            np.ndarray: shape (n1, n1 + n2)
            pd.Index: indices of unlabeled data
            pd.Index: indices of labeled data
        """
        ux_idx = list(u_x.index)
        # todo: get cache
        if self.__getattribute__("dist_cache_path") is not None:
            cache_path = self.__getattribute__("dist_cache_path")
            # debug
            # print(f"loading cache {cache_path}...")
            if self.__cache_d is None:
                self.__cache_d = np.load(cache_path)

            return self.__cache_d[np.array(ux_idx)[:, np.newaxis], np.array(ux_idx)[:, np.newaxis]], ux_idx

        # ux_idx = {i: j for i, j in enumerate(ux_idx)}
        # lx_idx = {i: j for i, j in enumerate(lx_idx)}
        _u_x = u_x.to_numpy()
        return self.__dist(u_x, u_x), ux_idx

    def __cluster(self, u_x: pd.DataFrame):
        d, ux_idx = self.dist(u_x)
        if len(ux_idx) < self.batch_size:
            # todo
            return list(range(len(ux_idx)))


    def __get_batch(self, u_x: pd.DataFrame, k: int):
        d, ux_idx = self.dist(u_x)
        batch = []
        r = np.array(list(range(len(ux_idx))))
        c = list(range(len(ux_idx)))
        c_max = len(c)
        for i in range(k):
            selected = d[r[:, np.newaxis], np.array(c)[np.newaxis, :]].min(axis=1).argmax()
            batch.append(selected)
            d[selected, :] = -np.inf
            c.append(selected + c_max)
        return batch


ALL_STRATEGIES = {
    "random": RandomStrategy,
    "uncertain": UncertainStrategy,
    "disagreement": DisagreementStrategy,
    "ranked_batch": RankedBatchModeStrategy,
    "information_density": InformationDensityStrategy,
    "gsx": GSxStrategy
}


def get_al_strategy(name: str, params: dict = None):
    if name in ALL_STRATEGIES:
        if params is None:
            return ALL_STRATEGIES[name]
        else:
            return ALL_STRATEGIES[name](**params)
    else:
        raise RuntimeError(
            f"Strategy {name} not exists. For custom strategy, please override the `ActiveLearningMetric` class"
        )
