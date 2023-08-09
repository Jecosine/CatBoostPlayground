import json
import os.path
import pickle

import pandas as pd

from allib.constants import *
from allib.utils import ensure_path, validate_dataset
from .tools import _download_dataset, _test_download

URL_TEMPLATE = "https://archive.ics.uci.edu/static/public/%s/%s.zip"
UCI_DB = None


# print(os.getcwd(), os.path.abspath(os.path.curdir))


def _load_raw_uci(raw_path: str = "./uci_db.json"):
    global UCI_DB
    db_cache_path = os.path.join(CACHE_DIR, "meta")
    ensure_path(db_cache_path)
    meta_path = os.path.join(db_cache_path, "uci_cache.pkl")
    # process meta data
    if UCI_DB is None:
        if not ensure_path(meta_path, is_dir=False):
            with open(raw_path, "rb") as f:
                raw_data = json.load(f)
            processed = {}
            for i in raw_data:
                processed[i["Name"].strip().lower().replace(" ", "-")] = i
            with open(meta_path, "wb") as f:
                pickle.dump(processed, f)
            UCI_DB = processed
        else:
            with open(meta_path, "rb") as f:
                UCI_DB = pickle.load(f)


def load_uci(
    name: str, reload: bool = False, test: bool = False, raw_path: str = "./uci_db.json"
):
    """ Load datasets from UCI repo by name

    Args:
        name: dataset name(lower case and replace space with '-')
        reload: reload if already exists
        test: do not perform real download operation
        raw_path: path to the crawled datasets info

    Returns:
        pd.DataFrame: the dataset in pandas.DataFrame form
    """
    _load_raw_uci(raw_path)
    name = name.lower()
    if name not in UCI_DB:
        # todo: new exception
        raise RuntimeError(f"Cannot find {name} in UCI Repo")
    dataset = UCI_DB[name]
    dataset_path = os.path.join(CACHE_DIR, name)
    if reload or validate_dataset(dataset_path):
        url = URL_TEMPLATE % (dataset["ID"], dataset["slug"])
        if not test:
            _download_dataset(url, "", name)
        else:
            _test_download(url, name)


def get_uci_db() -> dict:
    global UCI_DB
    return UCI_DB or {}


def iris(path: str):
    checklists = [os.path.join(path, i) for i in ["iris.data", "bezdekIris.data"]]
    data1 = pd.read_csv(checklists[0])
