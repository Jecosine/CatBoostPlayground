from copy import deepcopy
import os.path
from typing import List
from allib.datasets import Dataset
import pandas as pd
from allib.utils import ensure_path


DS_CACHE = "processed_datasets"

pp_names = [
    "continuous_to_categorical",
    "pca_3",
    # "pca_2",
]
pp_steps = [
    ["continuous_to_categorical"],
    ["categorical_to_continuous", "pca"],
    # ["categorical_to_continuous", "pca"]
]
pp_params_list = [
    [{"encode": "ordinal"}],
    [{}, {"n_components": 3}],
    # [{}, {"n_components": 2}]
]


def save(data: pd.DataFrame, label: pd.DataFrame, path: str):
    ensure_path(path)
    data.to_csv(os.path.join(path, "data.csv"), header=True, index=False)
    label.to_csv(os.path.join(path, "label.csv"), header=True, index=False)


def check(variance_name: str):
    path = os.path.join(DS_CACHE, variance_name)
    return sum([os.path.exists(os.path.join(path, fn)) for fn in ["data.csv", "label.csv"]]) == 2


def process_dataset(name: str, dataset: Dataset, reload: bool=False) -> dict:
    """ produce variances of datasets:
    1. pca_2
    2. pca_3
    3. cat

    Args:
        name: dataset name
        dataset: dataset to process

    Returns:
        List[Dataset]: processed datasets
    """
    data, label = dataset._data, dataset._label
    datasets = {}

    for pp_name, steps, params in zip(pp_names, pp_steps, pp_params_list):
        variance_name = f"{name}/{pp_name}"
        path = os.path.join(DS_CACHE, variance_name)
        if reload or not check(variance_name):
            print(f"[DATASET]: preprocessing {variance_name}")
            new_dataset = dataset.with_preprocess(
                steps=steps,
                params_list=params,
                in_place=False
            )
            save(new_dataset._data, label, path)
        else:
            new_dataset = deepcopy(dataset)
            print(f"[DATASET]: loading exist csv for {path}")
            new_dataset._data = pd.read_csv(os.path.join(path, "data.csv"), header=0)
            new_dataset._label = pd.read_csv(os.path.join(path, "label.csv"), header=0)
            if pp_name == "continuous_to_categorical":
                new_dataset._data = new_dataset._data.astype("category")
            new_dataset.reset()
        datasets[variance_name] = new_dataset
    return datasets


def load_datasets(origin_dataset_dict: dict):
    dataset_dict = {}
    for dataset_name, dataset in origin_dataset_dict.items():
        dataset_dict = dataset_dict | process_dataset(dataset_name, dataset, True)
    print("-----" * 6)
    print("Datasets loaded:")
    for name in dataset_dict:
        print(f"- {name}")
    print("-----" * 6)
    return dataset_dict
