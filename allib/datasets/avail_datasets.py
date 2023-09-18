import json
import logging
import os.path
import pickle
from importlib import resources
from typing import Optional

import pandas as pd

from allib.constants import *
from allib.utils import ensure_path, validate_dataset
from .Dataset import Dataset
from .tools import _download_dataset, _test_download

URL_TEMPLATE = "https://archive.ics.uci.edu/static/public/%s/%s.zip"
UCI_DB = None


__logger = logging.Logger(__name__)


def _load_raw_uci(raw_path: str = "uci_db.json"):
    global UCI_DB
    db_cache_path = os.path.join(CACHE_DIR, "meta")
    ensure_path(db_cache_path)
    meta_path = os.path.join(db_cache_path, "uci_cache.pkl")
    # process meta data
    if UCI_DB is None:
        if not ensure_path(meta_path, is_dir=False):
            # with open(raw_path, "rb") as f:
            #     raw_data = json.load(f)
            raw_data = resources.files("allib.datasets").joinpath(raw_path).read_text()
            raw_data = json.loads(raw_data)
            processed = {}
            for i in raw_data:
                processed[i["Name"].strip().lower().replace(" ", "-")] = i
            with open(meta_path, "wb") as f:
                pickle.dump(processed, f)
            UCI_DB = processed
        else:
            with open(meta_path, "rb") as f:
                UCI_DB = pickle.load(f)


def get_uci_db() -> dict:
    global UCI_DB
    _load_raw_uci()
    return UCI_DB or {}


def iris(path: Optional[str] = None):
    if not UCI_DB:
        _load_raw_uci()
    path = path or os.path.join(CACHE_DIR, "iris")
    checklists = [os.path.join(path, i) for i in ["iris.data", "bezdekIris.data"]]
    for p in checklists:
        ensure_path(p, is_dir=False)
    attrs = UCI_DB["iris"]["attributes"]
    columns = []
    for attr in attrs:
        columns.append(attr["name"].strip().lower().replace(" ", "_"))
    d1 = pd.read_csv(checklists[0], skiprows=0, names=columns)
    d2 = pd.read_csv(checklists[1], skiprows=0, names=columns)
    data = pd.concat((d1, d2)).reset_index(drop=True)
    # separate label column
    data = data.rename(columns={"class": "label"})
    label = data.label
    data = data.drop(columns=["label"])
    # return (data, label), (None, None), []
    return Dataset(
        data=data,
        label=label,
        al_metric=None,
        shuffle=False,
        init_size=30,
        batch_size=20,
    )


def adult(path: Optional[str] = None):
    if not UCI_DB:
        _load_raw_uci()
    path = path or os.path.join(CACHE_DIR, "adult")
    checklists = [os.path.join(path, i) for i in ["adult.data", "adult.test"]]
    for p in checklists:
        ensure_path(p, is_dir=False)
    attrs = UCI_DB["adult"]["attributes"]
    columns = []
    cat_idx = []
    for i, attr in enumerate(attrs):
        columns.append(attr["name"].strip().lower().replace("-", "_"))
        if attr["type"] in ["Categorical", "Binary"]:
            cat_idx.append(i)
    d1 = pd.read_csv(checklists[0], skiprows=0, names=columns)
    d2 = pd.read_csv(checklists[1], skiprows=1, names=columns)

    # separate label column
    data = d1.rename(columns={"income": "label"})
    label = data.label.apply(str.strip)  # remove space
    data = data.drop(columns=["label"])
    testset = d2.rename(columns={"income": "label"})
    test_y = testset.label.apply(lambda x: x.strip()[:-1])  # remove tail and space
    test_x = testset.drop(columns=["label"])
    # todo: batch size setting
    dataset = Dataset(
        data=data,
        label=label,
        al_metric=None,
        shuffle=False,
        init_size=30,
        batch_size=50,
    )
    dataset.info["cat_idx"] = cat_idx
    dataset.test_x, dataset.test_y = test_x, test_y
    return dataset


def yeast(path: Optional[str] = None):
    if not UCI_DB:
        _load_raw_uci()
    path = path or os.path.join(CACHE_DIR, "yeast")
    checklists = [os.path.join(path, i) for i in ["yeast.data", "yeast.names"]]
    for p in checklists:
        ensure_path(p, is_dir=False)
    attrs = UCI_DB["yeast"]["attributes"]
    columns = []
    cat_idx = []
    for i, attr in enumerate(attrs):
        columns.append(attr["name"].strip().lower().replace("-", "_"))
    d = pd.read_csv(checklists[0], skiprows=0, delim_whitespace=True, names=columns)
    # separate label column
    data = d.rename(columns={"localization_site": "label"})
    label = data.label.apply(str.strip)  # remove space
    data = data.drop(columns=["sequence_name", "label"])
    dataset = Dataset(
        data=data,
        label=label,
        al_metric=None,
        shuffle=False,
        init_size=30,
        batch_size=20,
    )
    dataset.info["cat_idx"] = cat_idx
    return dataset


def letter_recognition(path: Optional[str] = None):
    if not UCI_DB:
        _load_raw_uci()
    path = path or os.path.join(CACHE_DIR, "letter-recognition")
    checklists = [os.path.join(path, i) for i in [
        "letter-recognition.data",
        "letter-recognition.names",
        "letter-recognition.data.Z"]]
    for p in checklists:
        ensure_path(p, is_dir=False)
    attrs = UCI_DB["letter-recognition"]["attributes"]
    columns = []
    cat_idx = []
    for i, attr in enumerate(attrs):
        columns.append(attr["name"].strip().lower().replace("-", "_"))
    d = pd.read_csv(checklists[0], skiprows=0, names=columns)
    data = d.rename(columns={"lettr": "label"})
    label = data.label.apply(str.strip)  # remove space
    data = data.drop(columns=["label"])
    dataset = Dataset(
        data=data, label=label, al_metric=None, shuffle=False, init_size=30, batch_size=50,
    )
    dataset.info["cat_idx"] = cat_idx
    return dataset


def image_segmentation(path: Optional[str] = None):
    if not UCI_DB:
        _load_raw_uci()
    path = path or os.path.join(CACHE_DIR, "image-segmentation")
    checklists = [os.path.join(path, i) for i in [
        "segmentation.data",
        "segmentation.names",
        "segmentation.test"]]
    for p in checklists:
        ensure_path(p, is_dir=False)
    attrs = UCI_DB["image-segmentation"]["attributes"]
    columns = []
    cat_idx = []
    for i, attr in enumerate(attrs):
        columns.append(attr["name"].strip().lower().replace("-", "_"))
    d1 = pd.read_csv(
        "dataset_cache/image-segmentation/segmentation.data", skiprows=5, names=columns
    )
    d2 = pd.read_csv(
        "dataset_cache/image-segmentation/segmentation.test", skiprows=5, names=columns
    )
    data = d1.rename(columns={"class": "label"})
    label = data.label.apply(str.strip)
    data = data.drop(columns=["label"])
    test_x = d2.rename(columns={"class": "label"})
    test_y = test_x.label.apply(str.strip)
    test_x = test_x.drop(columns=["label"])
    dataset = Dataset(
        data=data, label=label, al_metric=None, shuffle=False, init_size=30, batch_size=20,
    )
    dataset.info["cat_idx"] = cat_idx
    dataset.test_x, dataset.test_y = test_x, test_y
    return dataset


AVAIL_DATASET = {
    "iris"               : iris,
    "adult"              : adult,
    "yeast"              : yeast,
    "letter-recognition" : letter_recognition,
    "image-segmentation" : image_segmentation
}


def load_uci(
    name: str, reload: bool = False, test: bool = False, raw_path: str = "uci_db.json"
) -> Dataset:
    """ Load datasets from UCI repo by name

    Args:
        name: dataset name(lower case and replace space with '-')
        reload: reload if already exists
        test: do not perform real download operation
        raw_path: path to the crawled datasets info

    Returns:
        pd.DataFrame: the dataset in pandas.DataFrame form
    """
    if not UCI_DB:
        _load_raw_uci(raw_path)
    name = name.lower()
    if name not in UCI_DB:
        # todo: new exception
        raise RuntimeError(f"Cannot find {name} in UCI Repo")
    dataset = UCI_DB[name]
    dataset_path = os.path.join(CACHE_DIR, name)
    # still reload if not valid
    if reload or not validate_dataset(dataset_path):
        url = URL_TEMPLATE % (dataset["ID"], dataset["slug"])
        if not test:
            _download_dataset(url, "", name)
        else:
            _test_download(url, name)

    if name not in AVAIL_DATASET:
        __logger.warning(
            f"Dataset {name} downloaded, but it need to be manually processed"
        )
    else:
        handler = AVAIL_DATASET[name]
        return handler()
