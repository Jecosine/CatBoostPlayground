import os
import pickle
# os.environ["OMP_NUM_THREADS"] = "1"
# ignore numpy warning
import warnings

import matplotlib.pyplot as plt
import numpy as np

from allib.datasets import AVAIL_DATASETS, load_uci
from allib.metrics import distance
from allib.plots import PLMetric
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')
BASE_PATH = "app/comp_dist/"
SUFFIX = "200iter_catonly_wo_n"
ENCODE = "ordinal_wo_n"
ALSTRATEGY = "typiclust"
candidates = ["euclidean", "cosine", "hamming", "smirnov", "goodall1", "iof", "anderberg", "gambaryan"]
def get_post_plot(m, s, pl_metric):
    def post_plot(metric_name, instances, metrics_n_times, strategies, ax, fig):
        for idx, metrics in enumerate([m[0]]):
            # med = np.median(metrics, axis=0)
            # mx = metrics.max(axis=0)
            # mn = metrics.min(axis=0)
            # q1 = np.quantile(metrics, 0.25, axis=0)
            # q3 = np.quantile(metrics, 0.75, axis=0)
            med, mx, mn = pl_metric.errorbar("std", metrics)
            ax.fill_between(instances, mn, mx, alpha=0.1, color='black')
            ax.plot(instances, med, label=f"baseline(random)", marker="o", linestyle="dashed", markersize=5, c='black')
        for idx, metrics in enumerate([m[1]]):
            med, mx, mn = pl_metric.errorbar("std", metrics)
            ax.fill_between(instances, mn, mx, alpha=0.1, color='grey')
            ax.plot(instances, med, label=f"random_dist", marker="o", linestyle="dashed", markersize=5, c='grey')
        ax.legend()
    return post_plot


def plot(
        dataset_name: str,
        model_name: str,
        # strategies: list = None,
        distance_metrics: list=None,
        plot_name: str = None,
        strategy: str = "gsx",
        encode: str = "ordinal"
):
    # load dataset
    ds = load_uci(dataset_name)
    # ds.with_preprocess(steps=["sample_n", "categorical_only", "remove_constant_columns", "drop_duplicate_rows"],  params_list=[{"n": 1000, "random_state": 0}, {}, {}, {}], in_place=True)
    
    ds.with_preprocess(steps=["sample_n", "remove_constant_columns", "drop_duplicate_rows"],  params_list=[{"n": 1000, "random_state": 0}, {}, {}], in_place=True)

    metrics_n_times = []
    instances = []
    distance_metrics = distance_metrics or list(distance.AVAIL_DIST_METRICS.keys())
    plot_name = plot_name if plot_name is not None else f"{dataset_name}@{model_name}@{strategy}@{encode}@{SUFFIX}_binary.png"
    # if ensure_path(os.path.join("./plots/pl_metric", plot_name), False):
    #     print(f"plot {plot_name} already exists, continue")
    #     return
    # for strategy in strategies:
    print(f"plotting {plot_name}...")
    for metric in distance_metrics:
        stats = None
        cache_name = f"{dataset_name.replace('/', '_')}@{model_name}@{strategy}_{metric}_{encode}@{SUFFIX}@x20.pkl"
        print("Loading " + os.path.join(BASE_PATH + "ppl_cache_pred", cache_name))
        if not os.path.isfile(os.path.join(BASE_PATH + "ppl_cache_pred", cache_name)):
            print(f"exp {os.path.join(BASE_PATH + 'ppl_cache_pred', cache_name)} does not exist, continue")
            continue
        with open(os.path.join(BASE_PATH + "ppl_cache_pred", cache_name), "rb") as f:
            stats = pickle.load(f)
        # print(len(stats[0]["predictions"]))
        metrics_n_times.append([[f1_score(ds.test_y, stats[i]["predictions"][j], pos_label="<=50K") for j in range(len(stats[i]["instances"]))] for i in range(len(stats))])
        # metrics_n_times.append([[f1_score(ds.test_y, stats[i]["predictions"][j], average="micro") for j in range(len(stats[i]["instances"]))] for i in range(len(stats))])
        # print(metrics_n_times)
        # metrics_n_times.append([stats[i]["accuracy"] for i in range(len(stats))])
        instances = stats[0]["instances"]
    baseline_euclidean = None
    baseline_cosine = None
    for metric in ["cosine", "random"]:
        stats = None
        if metric == "cosine":
            cache_name = f"{dataset_name.replace('/', '_')}@{model_name}@random_{metric}_{encode}@{SUFFIX}@x20.pkl"
        else:
            cache_name = f"{dataset_name.replace('/', '_')}@{model_name}@{strategy}_{metric}_{encode}@{SUFFIX}@x20.pkl"

        if not os.path.isfile(os.path.join(BASE_PATH + "ppl_cache_pred", cache_name)):
            print(f"exp {cache_name} does not exist, continue")
            continue
        print(os.path.join(BASE_PATH + "ppl_cache_pred", cache_name))
        with open(os.path.join(BASE_PATH + "ppl_cache_pred", cache_name), "rb") as f:
            stats = pickle.load(f)
        metrics_n_times.append([[f1_score(ds.test_y, stats[i]["predictions"][j], pos_label="<=50K") for j in range(len(stats[i]["instances"]))] for i in range(len(stats))])
        # metrics_n_times.append([[f1_score(ds.test_y, stats[i]["predictions"][j], average="micro") for j in range(len(stats[i]["instances"]))] for i in range(len(stats))])
    print(len(metrics_n_times))
    for i in metrics_n_times:
        print(",".join([str(len(j)) for j in i]))
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # ax.set_ylim(0.8, 0.88)

    pl_metric = PLMetric(output_path="app/comp_dist/plots/100-iter-f1/", ax=ax)
    pl_metric.plot(
        metric_name="F1-Binary",
        instances=instances,
        metrics_n_times=np.array(metrics_n_times[:-2]),
        strategies=distance_metrics,
        plot_name=plot_name,
        cmap=plt.get_cmap("hsv"),
        dpi=300,
        title=f"F1 {dataset_name}@{model_name}@{strategy}",
        errorbar="std",
        post_plot=get_post_plot(np.array(metrics_n_times[-2:]), ["cosine", "random"], pl_metric)
    )
    pl_metric.savefig(plot_name)

plt.rcParams.update({'font.size': 16})
for dsn in ["adult"]:
# for dsn in AVAIL_DATASETS:
    # if dsn == 'yeast':
    #     continue
    # plot(dsn, "catboost", distance_metrics=list(["of", "goodall4", "overlap", "smirnov"]), strategy="typiclust")
    # plot(dsn, "catboost", distance_metrics=list(distance.AVAIL_DIST_METRICS.keys()), strategy=ALSTRATEGY, encode="ordinal_wo_n")
    plot(dsn, "catboost", distance_metrics=candidates, strategy=ALSTRATEGY, encode="ordinal_wo_n", plot_name=f"{dsn}@catboost@{ALSTRATEGY}@{ENCODE}@{SUFFIX}_thesis.png")
    
