from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.stats import entropy, norm

from allib.typing import ArrayLike
from allib.utils import arg_bottomk, arg_topk, get_dist_metric


class ActiveLearningMetric(ABC):
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


class UncertainMetric(ActiveLearningMetric):
    def __init__(
        self, name: str="uncertainty", *args, **kwargs,
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


class DisagreementMetric(ActiveLearningMetric):
    def __init__(self, models: list, name: str = "vote", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.models = models
        self.func_strategy = {
            "vote": self.__func_vote,
            "consensus": self.__func_consensus,
            "max_disagreement": self.__func_max_disagreement,
        }[self.name]

    def __func_vote(self, pred: ArrayLike):
        res = pred.argmax(axis=-1).T
        n_votes = res.shape[-1]
        votes = np.zeros(res.shape)
        for i in range(votes.shape[0]):
            unique, count = np.unique(res, return_counts=True)
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
        n_samples, n_votes = res.shape
        kl = np.zeros((n_samples, n_votes))
        for i in range(n_samples):
            for j in range(n_votes):
                kl[i, j] = entropy(pred[j, i], qk=c_e[i])
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
        models = kwargs.get("models", None)
        if models is None:
            raise RuntimeError(f"Disagreement metric requires a `models` parameter")
        if initial_batch:
            return self.sample_initial(train_x, train_y, random_state)
        if len(train_x) > self.batch_size:
            pred = np.array(
                [model.predict_proba(train_x).max(axis=1) for model in models]
            )
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


class RankedBatchModeMetric(ActiveLearningMetric):
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
        alpha: float = kwargs.get("alpha", None)
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


class InformationDensityMetric(ActiveLearningMetric):
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
            info = self.sim(train_x, train_x).sum(axis=1).mean(axis=1)
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


class AcquisitionMetric(ActiveLearningMetric):
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


__ALL_METRICS = {
    "random": RandomMetric,
    "uncertain": UncertainMetric,
    "disagreement": DisagreementMetric,
    "ranked_batch": RankedBatchModeMetric,
    "information_density": InformationDensityMetric,
}


def get_al_metric(name: str, params: dict):
    if name in __ALL_METRICS:
        return __ALL_METRICS[name](**params)
    else:
        raise RuntimeError(
            f"Metric {name} not exists. For custom metric, please override the `ActiveLearningMetric` class"
        )
