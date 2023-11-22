from typing import List
import pickle
import os

def various_preprocesses(dataset_name, model_name, al_confs, al_strategies):
    metrics_n_times = []
    instances = []
    strategies = []
    plot_name = f"{dataset_name}@{model_name}.png"
    for idx, confs in enumerate(al_confs):
        for cid, conf in enumerate(confs):
            stats = None
            cache_name = f"{dataset_name}@{model_name}@{al_strategies[idx]}_{cid}@x20.pkl"
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
