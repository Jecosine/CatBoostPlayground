import warnings
from typing import List
from operator import itemgetter
import pandas as pd
import numpy as np
from pandas import CategoricalDtype
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import PCA
from .tools import remove_inf
# todo: kbins return inf which will be processed as one of the category


def categorical_only(data: pd.DataFrame, label: pd.DataFrame = None):
    columns = data.columns
    cats = []
    for col in columns:
        if not isinstance(data[col].dtype, CategoricalDtype):
            cats.append(col)
    if len(cats) == 0:
        warnings.warn("[DATASET]: No categorical columns found")
    return data[cats].astype("category"), label


def continuous_only(data: pd.DataFrame, label: pd.DataFrame = None):
    columns = data.columns
    conts = []
    for col in columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            conts.append(col)
    if len(conts) == 0:
        warnings.warn("[DATASET]: No continuous value columns found")
    return data[conts], label


def continuous_to_categorical(data: pd.DataFrame, label: pd.DataFrame, params: dict = None):
    params = params or {}
    kbin = KBinsDiscretizer(**params)
    # get continuous cols
    columns = data.columns
    num_idx = []
    for col in columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            num_idx.append(col)

    if len(num_idx) == 0:
        return data, label
    # kbin.fit(data[num_idx])
    # data[num_idx] = kbin.transform(data[num_idx])
    discret_data = kbin.fit_transform(data[num_idx])
    discret_data = pd.DataFrame(discret_data, columns=kbin.get_feature_names_out(), index=data.index)
    # concat the data by column
    # drop columns by column id
    data = data.drop(columns=num_idx)
    data = pd.concat([data, discret_data], axis=1)
    data["label"] = label
    data = remove_inf(data)
    label = data["label"]
    data = data.drop(columns=["label"])
    # convert every column to category
    for col in data.columns:
        if col == "label":
            continue
        data[col] = data[col].astype("category").cat.codes
        data[col] = data[col].astype(int).astype("category")

    return data, label


def categorical_to_continuous(data: pd.DataFrame, label: pd.DataFrame):
    # get category cols
    columns = data.columns
    cat_idx = []
    for col in columns:
        if pd.api.types.is_categorical_dtype(data[col]):
            cat_idx.append(col)
    if len(cat_idx) == 0:
        return data
    return pd.get_dummies(data), label


def pca(data: pd.DataFrame, label: pd.DataFrame, params: dict):
    do_pca = PCA(**params)
    do_pca.fit(data)
    return pd.DataFrame(do_pca.fit_transform(data)), label


def sample_n(data: pd.DataFrame, label: pd.DataFrame, params: dict):
    n = params.get("n", 100)
    random_state = params.get("random_state", None)
    if n is None:
        raise RuntimeError("Sample n is not specified")
    data = data.sample(n=min(n, data.shape[0]), random_state=random_state)
    label = label.loc[data.index]
    data = data.reset_index(drop=True)
    label = label.reset_index(drop=True)
    return data, label


def remove_constant_columns(data: pd.DataFrame, label: pd.DataFrame):
    return data.loc[:, data.nunique() != 1], label


ALL_PREPROCESSES = {
    "categorical_only": categorical_only,
    "continuous_only": continuous_only,
    "continuous_to_categorical": continuous_to_categorical,
    "categorical_to_continuous": categorical_to_continuous,
    "pca": pca,
    "sample_n": sample_n,
    "remove_constant_columns": remove_constant_columns,
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

    def ppl(data: pd.DataFrame, label: pd.DataFrame = None):
        if params_list and (len(params_list) != len(ppss)):
            raise RuntimeError("Params list length does not match the step number")
        for pps, params in zip(ppss, params_list):
            if not params:
                data, label = pps(data, label)
            else:
                data, label = pps(data, label, params)
        return data, label

    return ppl
