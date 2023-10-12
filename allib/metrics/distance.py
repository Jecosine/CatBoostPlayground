from typing import List

import numpy as np

from allib.typing import ArrayLike

# todo:
#  1. filter not work if equal
#  2. def of q
#  3. rewrite in np.select

def __make_ldict_getter(d: List[dict]):
    def wrapper(idx, val):
        return d[idx].get(val)
    return wrapper


def __m_overlap(X: ArrayLike, Y: ArrayLike):
    """
    Args:
        X: of shape (n_features,)
        Y: of shape (n_features.)

    Returns:
        scalar
    """
    return np.array(X == Y).mean()


def __m_eskin(X: ArrayLike, Y: ArrayLike, nks: ArrayLike):
    """
    Args:
        X: of shape (n_features,)
        Y: of shape (n_features,)

    Returns:
        scalar
    """
    nks = np.power(nks, 2)
    nks = nks/(nks + 2.0)
    eq = np.array(X == Y)
    res = eq.astype(float)
    res[~eq] = nks[~eq]
    return res.mean()


def __m_iof(X: ArrayLike, Y: ArrayLike, freq: List[dict]):
    """
    Args:
        X: of shape (n_features,)
        Y: of shape (n_features,)
        freq: dict of {feature: {value: frequency}}

    Returns:
        scalar
    """
    freq_getter = __make_ldict_getter(freq)
    indices = np.arange(X.shape[0])
    xf = np.vectorize(freq_getter)(indices, X)
    yf = np.vectorize(freq_getter)(indices, Y)
    freq = 1.0 / (1.0 + np.multiply(np.log(xf), np.log(yf)))
    eq = np.array(X == Y)
    res = eq.astype(float)
    res[~eq] = freq[~eq]
    return res.mean()


def __m_of(X: ArrayLike, Y: ArrayLike, freq: List[dict], N: int):
    """
    Args:
        X: of shape (n_features,)
        Y: of shape (n_features,)
        freq: dict of {feature: {value: frequency}}
        N: size of dataset

    Returns:
        float
    """
    N: float = 1.0 * N
    freq_getter = __make_ldict_getter(freq)
    xf = np.vectorize(freq_getter)(X)
    yf = np.vectorize(freq_getter)(Y)
    freq = 1.0 / (1.0 + np.multiply(np.log(N / xf), np.log(N / yf)))
    eq = np.array(X == Y)
    res = eq.astype(float)
    res[~eq] = freq[~eq]
    return res.mean()


def __m_lin(X: ArrayLike, Y: ArrayLike, prob: List[dict]):
    """
    Args:
        X: of shape (n_features,)
        Y: of shape (n_features,)
        prob: dict of {feature: {value: probability}}
            $p_k(x) = \frac{f_k(x)}{N}$

    Returns:
        float
    """
    prob_getter = __make_ldict_getter(prob)
    xp = np.vectorize(prob_getter)(X)
    yp = np.vectorize(prob_getter)(Y)
    otherwise = 2.0 * np.log(xp + yp)
    eq = np.array(X == Y)
    res = eq.astype(float) * 2.0 * np.log(xp)
    res[~eq] = otherwise[~eq]
    w = 1.0 / ((np.log(xp) + np.log(yp)).sum())
    # shape check
    d = X.shape[0]
    assert w.shape[0] == d
    return res.sum() * w


def __make_array_filter(arr: List, episilon: float = 1e-7):
    def wrapper(i, a=None, b=None):
        if a is None and b is None:
            raise RuntimeError("min and max cannot be both None")
        carr = arr[i]
        if a is None:
            a = carr.min()
        if b is None:
            b = carr.max()
        a, b = (a, b) if a <= b else (b, a)
        return carr[(carr >= a - episilon) & (carr <= b + episilon)]
    return wrapper


def __m_lin1(X: ArrayLike, Y: ArrayLike, prob: List[dict]):
    """
    Args:
        X: of shape (n_features,)
        Y: of shape (n_features,)
        prob: list of dict of {feature: {value: probability}}
            $p_k(x) = \frac{f_k(x)}{N}$

    Returns:
        float
    """
    ap = [np.array(p.values()) for p in prob]
    indices = np.arange(X.shape[0])
    prob_getter = __make_ldict_getter(prob)
    array_filter = __make_array_filter(ap)
    xp = np.vectorize(prob_getter)(X)
    yp = np.vectorize(prob_getter)(Y)
    clip_array = np.frompyfunc(
        array_filter,
        3, 1)
    clipped = clip_array(indices, xp, yp)
    qs = np.array([(np.log(c).sum(), 2 * np.log(c.sum())) for c in clipped]).T
    eq = np.array(X == Y)
    # if eq
    res = eq.astype(float) * qs[0]
    # otherwise
    res[~eq] += qs[1, ~eq]
    w = 1.0 / (qs[0].sum())
    # shape check
    d = X.shape[0]
    assert w.shape[0] == d
    return res.sum() * w


def __m_goodall1(X: ArrayLike, Y: ArrayLike, prob2: List[dict]):
    """
    Args:
        X: of shape (n_features,)
        Y: of shape (n_features,)
        prob2: list of dict of {feature: {value: probability}}
            $p^2_k(x) = \frac{f_k(x)(f_k(x) - 1)}{N(N - 1)}$

    Returns:
        float
    """
    prob2_getter = __make_ldict_getter(prob2)
    indices = np.arange(X.shape[0])
    xp = np.vectorize(prob2_getter)(indices, X)
    ap = [np.array(p.values()) for p in prob2]
    array_filter = __make_array_filter(ap)
    clip_array = np.frompyfunc(
        array_filter,
        3, 1)
    clipped = clip_array(indices, None, xp)
    # todo optimize
    qs = np.array(
        [
            c.sum()
            for c in clipped
        ]
    )
    eq = np.array(X == Y)
    res = eq.astype(float) * (1 - qs)
    return res.mean()


def __m_goodall2(X: ArrayLike, Y: ArrayLike, prob2: List[dict]):
    """
    Args:
        X: of shape (n_features,)
        Y: of shape (n_features,)
        prob2: list of dict of {feature: {value: probability}}
            $p^2_k(x) = \frac{f_k(x)(f_k(x) - 1)}{N(N - 1)}$

    Returns:
        float
    """
    indices = np.arange(X.shape[0])
    prob2_getter = __make_ldict_getter(prob2)
    xp = np.vectorize(prob2_getter)(indices, X)
    ap = [np.array(p.values()) for p in prob2]
    array_filter = __make_array_filter(ap)
    clip_array = np.frompyfunc(
        array_filter,
        3, 1)
    clipped = clip_array(indices, xp, None)
    # todo optimize
    qs = np.array(
        [
            c.sum()
            for c in clipped
        ]
    )
    eq = np.array(X == Y)
    res = eq.astype(float) * (1 - qs)
    return res.mean()


def __m_goodall3(X: ArrayLike, Y: ArrayLike, prob2: List[dict]):
    """
    Args:
        X: of shape (n_features,)
        Y: of shape (n_features,)
        prob2: list of dict of {feature: {value: probability}}
            $p^2_k(x) = \frac{f_k(x)(f_k(x) - 1)}{N(N - 1)}$

    Returns:
        float
    """
    prob2_getter = __make_ldict_getter(prob2)
    indices = np.arange(X.shape[0])
    xp2 = np.vectorize(prob2_getter)(indices, X)
    eq = np.array(X == Y)
    res = eq.astype(float) * (1 - xp2)
    return res.mean()


def __m_goodall4(X: ArrayLike, Y: ArrayLike, prob2: dict):
    """
    Args:
        X: of shape (n_features,)
        Y: of shape (n_features,)
        prob2: list of dict of {feature: {value: probability}}
            $p^2_k(x) = \frac{f_k(x)(f_k(x) - 1)}{N(N - 1)}$

    Returns:
        float
    """
    prob2_getter = __make_ldict_getter(prob2)
    indices = np.arange(X.shape[0])
    xp2 = np.vectorize(prob2_getter)(indices, X)
    eq = np.array(X == Y)
    res = eq.astype(float) * xp2
    return res.mean()


def __m_smirnov(X: ArrayLike, Y: ArrayLike, freq: List[dict], N: int, nks: ArrayLike):
    """
    Args:
        X: of shape (n_features,)
        Y: of shape (n_features,)
        freq: list of dict of {feature: {value: frequency}}
        N: size of dataset
        nks: of shape (n_features,) number of possible values for each feature

    Returns:
        float
    """
    freq_getter = __make_ldict_getter(freq)
    indices = np.arange(X.shape[0])
    xf = np.vectorize(freq_getter)(indices, X)
    yf = np.vectorize(freq_getter)(indices, Y)
    calc = lambda arr: ((N - arr) / arr).sum()
    aq = np.array([calc(np.array(f.values())) for f in freq])
    xq = (N - xf) / xf
    yq = (N - yf) / yf
    i_aqs = (1.0 / aq).sum()
    eq = np.array(X == Y)
    res = eq.astype(float) * (xq + i_aqs - (1.0 / xq))

    res[~eq] = -2.0 + i_aqs - (1.0 / xq[~eq]) - (1.0 / yq[~eq])

    w = 1.0 / nks.sum()
    return res.sum() * w


def __m_gambaryan(X: ArrayLike, Y: ArrayLike, prob: List[dict], nks: ArrayLike):
    """
    Args:
        X: of shape (n_features,)
        Y: of shape (n_features,)
        prob: list of dict of {feature: {value: probability}}
        nks: of shape (n_features,) number of possible values for each feature

    Returns:
        float
    """
    # todo need to check prob2 range
    prob_getter = __make_ldict_getter(prob)
    indices = np.arange(X.shape[0])
    xp = np.vectorize(prob_getter)(indices, X)

    eq = np.array(X == Y)
    res = eq.astype(float) * (-xp * np.log2(xp) + (1 - xp) * (np.log2(1 - xp)))
    w = 1.0 / nks.sum()
    return res.sum() * w


def __m_burnaby(X: ArrayLike, Y: ArrayLike, prob: List[dict]):
    """
    Args:
        X: of shape (n_features,)
        Y: of shape (n_features,)
        prob: dict of {feature: {value: probability}}

    Returns:
        float
    """
    prob_getter = __make_ldict_getter(prob)
    indices = np.arange(X.shape[0])
    xp = np.vectorize(prob_getter)(indices, X)
    yp = np.vectorize(prob_getter)(indices, Y)
    aps = np.array(
        [2.0 * np.log(1.0 - np.array(p.values())).sum() for p in prob]
    )

    eq = np.array(X == Y)
    res = eq.astype(float)
    otherwise = aps / (np.log(xp * yp / (1.0 - xp) / (1.0 - yp)) + aps)
    res[~eq] = otherwise[~eq]
    return res.mean()


def __m_anderberg(X: ArrayLike, Y: ArrayLike, prob: List[dict], nks: ArrayLike):
    """
    Args:
        X: of shape (n_features,)
        Y: of shape (n_features,)
        prob: dict of {feature: {value: probability}}

    Returns:
        float
    """
    prob_getter = __make_ldict_getter(prob)
    indices = np.arange(X.shape[0])
    xp = np.vectorize(prob_getter)(indices, X)
    yp = np.vectorize(prob_getter)(indices, Y)

    eq = np.array(X == Y)
    res = eq.astype(float)
    # eq_res, neq_res = res[eq], res[~eq]
    eq_xp, neq_xp, neq_yp = xp[eq], xp[~eq], yp[~eq]
    eq_nks, neq_nks = nks[eq], nks[~eq]
    eq_res = (np.power(1.0 / eq_xp, 2) * 2.0 / (eq_nks * (eq_nks + 1.0))).sum()
    neq_res = (1.0 / (2.0 * neq_xp * neq_yp) * 2.0 / (eq_nks * (eq_nks + 1.0))).sum()

    return eq_res / (eq_res + neq_res)


# https://conservancy.umn.edu/bitstream/handle/11299/215736/07-022.pdf?sequence=1&isAllowed=y
_AVAIL_CAT_METRICS = [
    "overlap",
    "eskin",
    "iof",
    "of",
    "lin",
    "lin1",
    "goodall1",
    "goodall2",
    "goodall3",
    "goodall4",
    "smirnov",
    "gambaryan",
    "burnaby",
    "anderberg"
]

_AVAIL_METRICS = [
    "euclidean",
    "cosine",
    *_AVAIL_CAT_METRICS
]
