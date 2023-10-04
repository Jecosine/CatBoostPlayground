import warnings
from typing import List
from operator import itemgetter
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import PCA


def categorical_only(data: pd.DataFrame):
    columns = data.columns
    cats = []
    for col in columns:
        if pd.api.types.is_categorical_dtype(data[col]):
            cats.append(col)
    if len(cats) == 0:
        warnings.warn("[DATASET]: No categorical columns found")
    return data[cats].astype("category")


def continuous_only(data: pd.DataFrame):
    columns = data.columns
    conts = []
    for col in columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            conts.append(col)
    if len(conts) == 0:
        warnings.warn("[DATASET]: No continuous value columns found")
    return data[conts]


def continuous_to_categorical(data: pd.DataFrame, params: dict = None):
    params = params or {}
    kbin = KBinsDiscretizer(**params)
    # get continuous cols
    columns = data.columns
    num_idx = []
    for col in columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            num_idx.append(col)
    if len(num_idx) == 0:
        return data
    kbin.fit(data[num_idx])
    data[num_idx] = kbin.transform(data[num_idx])
    data[num_idx] = data[num_idx].astype(int).astype("category")
    return data


def categorical_to_continuous(data: pd.DataFrame):
    # get category cols
    columns = data.columns
    cat_idx = []
    for col in columns:
        if pd.api.types.is_categorical_dtype(data[col]):
            cat_idx.append(col)
    if len(cat_idx) == 0:
        return data
    return pd.get_dummies(data)


def pca(data: pd.DataFrame, params: dict):
    pca = PCA(**params)
    pca.fit(data)
    return pd.DataFrame(pca.fit_transform(data))


ALL_PREPROCESSES = {
    "categorical_only": categorical_only,
    "continuous_only": continuous_only,
    "continuous_to_categorical": continuous_to_categorical,
    "categorical_to_continuous": categorical_to_continuous,
    "pca": pca
}


def get_preprocess(name: str):
    pps = ALL_PREPROCESSES.get(name)
    if pps is None:
        raise RuntimeError(f"Preprocess {name} is not implemented")
    return pps


def build_preprocess_ppl(steps: List[str], params_list: List[dict]):
    ppss = itemgetter(*steps)(ALL_PREPROCESSES)
    if not isinstance(ppss, tuple):
        ppss = (ppss, )

    def ppl(data: pd.DataFrame):
        if params_list and (len(params_list) != len(ppss)):
            raise RuntimeError("Params list length does not match the step number")
        for pps, params in zip(ppss, params_list):
            if not params:
                data = pps(data)
            else:
                data = pps(data, params)
        return data

    return ppl
