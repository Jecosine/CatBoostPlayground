import os.path
import pickle
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import faulthandler
from allib.datasets import load_uci, AVAIL_DATASETS, Dataset
from allib.models import get_pipeline, AVAIL_MODELS
from allib.models.al import ActiveLearningStrategy, ActiveLearningPipeline, get_al_strategy, ALL_STRATEGIES
from allib.metrics import get_metrics
from allib.plots import PLMetric
from allib.utils import make_seeds, ensure_path
from preprocess import process_dataset, load_datasets


DS_CACHE = "processed_datasets"

# ---------------- MODEL ---------------
cat_models = [
    "catboost",
]
all_models = [
    # "logistic",
    # "mlp",
    "random_forest",
    "svm",
    *cat_models
]
model_dict = {
    name: get_pipeline(name) for name in all_models
}

# ---------------- DATASET ---------------
# all_datasets = list(AVAIL_DATASETS.keys())
all_preprocesses = [
    # "continuous_to_categorical"
    "pca_3",
    # "pca_2",
    # "origin"
]
all_datasets = [f"{ds}/{pps}" for ds in AVAIL_DATASETS.keys() for pps in all_preprocesses]

origin_dataset_dict = {
    name: load_uci(name) for name in AVAIL_DATASETS.keys()
}

dataset_dict = load_datasets(origin_dataset_dict)

# ---------------- STRATEGY ---------------
cat_strategies = [
    "random",
    "uncertain",
    "disagreement",
]

al_strategies =[
    *cat_strategies,
    "ranked_batch",
    "information_density",
    "gsx"
]

cat_confs = [
    [{}],
    [{"name": name} for name in ["uncertainty", "margin", "entropy"]],
    [{"name": name} for name in ["vote", "consensus", "max_disagreement"]],
]
al_confs = [
    *cat_confs,
    # todo
    [{"dist_metric": metric} for metric in ["cosine", "euclidean"]],
    [{"similarity_metric": metric} for metric in ["cosine", "euclidean"]],
    [{}]
]


def run_pipeline(dataset_name: str, model_name: str):
    origin_dataset = dataset_dict[dataset_name]
    avail_strategies = al_strategies
    avail_confs = al_confs
    if origin_dataset.info["cat_idx"] and len(origin_dataset.info["cat_idx"]) != 0:
        avail_strategies = cat_strategies
        avail_confs = cat_confs
    for idx, confs in enumerate(avail_confs):
        for cid, conf in enumerate(confs):
            cache_name = f"{dataset_name.replace('/', '_')}@{model_name}@{avail_strategies[idx]}_{cid}@x20.pkl"
            print(f"[PPL]: Checking {cache_name} ... ", end="")
            if os.path.isfile(os.path.join("ppl_cache", cache_name)):
                print(f" exists.")
                # continue
            else:
                strategy = get_al_strategy(avail_strategies[idx])
                if avail_strategies[idx] == "disagreement":
                    conf["make_model"] = AVAIL_MODELS[model_name]._model_maker
                dataset = origin_dataset.with_strategy(strategy, conf)
                make_ppl: Type[ActiveLearningPipeline] = model_dict[model_name]
                print(f"\nTraining pipeline: {model_name}; Dataset: {dataset_name}; AL Strategy: {avail_strategies[idx]}...")
                ppl = make_ppl(
                    model=None,
                    eval_metrics=get_metrics(["accuracy"]),
                    seeds=[i for i in range(20)],
                    n_times=20,
                    dataset=dataset,
                    cat_idx=dataset.info["cat_idx"]
                )
                ppl.start()
                with open(os.path.join("ppl_cache", cache_name), "wb") as f:
                    pickle.dump(ppl.stats, f)


def plot(dataset_name: str, model_name: str):
    metrics_n_times = []
    instances = []
    strategies = []
    plot_name = f"{dataset_name}@{model_name}.png"
    for idx, confs in enumerate(al_confs):
        for cid, conf in enumerate(confs):
            stats = None
            cache_name = f"{dataset_name.replace('/', '_')}@{model_name}@{al_strategies[idx]}_{cid}@x20.pkl"
            if not os.path.isfile(os.path.join("ppl_cache", cache_name)):
                continue
            with open(os.path.join("ppl_cache", cache_name), "rb") as f:
                stats = pickle.load(f)
            metrics_n_times.append([stats[i]["accuracy"] for i in range(len(stats))])
            strategies.append(f"{al_strategies[idx]}{('(' + list(conf.values())[0] + ')') if conf else ''}")
            instances = stats[0]["instances"]
    pl_metric = PLMetric()
    pl_metric.plot("Accuracy", instances, np.array(metrics_n_times), strategies, plot_name=plot_name)
    # pl_metric = PLMetric("Accuracy", instances, metrics_n_times, strategies, plot_name=plot_name)


def dataset_distribution(dataset_name: str):
    dataset = dataset_dict[dataset_name]
    title = f"Distribution of dataset {dataset_name}"
    fig, ax = plt.subplots(figsize=(10, 10))
    x1, x2 = dataset._data.x1, dataset._data.x2
    label = dataset._label
    ax.scatter(x1, x2, c=label.label.astype("category").cat.codes)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.savefig(f"plots/datasets/{dataset_name}.png")


if __name__ == "__main__":
    faulthandler.enable()
    for dsn, dataset in dataset_dict.items():
        avail_models = all_models
        if dataset.info["cat_idx"] and len(dataset.info["cat_idx"]) != 0:
            print("[MAIN] Checkout to cat models")
            avail_models = cat_models
        for mdn in avail_models:
            # run_pipeline(dsn, mdn)
            plot(dsn, mdn)
