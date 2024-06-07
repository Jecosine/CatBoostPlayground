import os 
#if you want to know current working dir
os.getcwd()
os.chdir('/home/jecosine/Courses/Courses/COMP8800/Projects/CatBoostPlayground')
os.listdir()

from allib.datasets import load_uci, AVAIL_DATASETS
import pandas as pd
import numpy as np
from allib.models import get_pipeline, AVAIL_MODELS
from allib.models.al import get_al_strategy
from allib.metrics import get_metrics
import pickle
import matplotlib.pyplot as plt
from allib.plots import PLMetric
from allib.metrics import distance
from allib.utils import ensure_path

RP = "/home/jecosine/Courses/Courses/COMP8800/Projects/CatBoostPlayground/examples"

make_ppl = get_pipeline('catboost')
ENCODE = 'ordinal'
DIST_CACHE = "examples/dist_cache"
LOCAL_DIR = "app/comp_bs"
ALSTRATEGY = "random"
BS = 1

SUFFIX = f"bs{BS}"
model_extra_params = { "iterations": 100 }
TOTAL = 100
# for dsn in AVAIL_DATASETS:
for dsn in ["adult"]:
    # if dsn in ["balance-scale"]:
    #     continue
    dataset = load_uci(dsn)
    dataset.with_preprocess(steps=["sample_n", "continuous_to_categorical", "remove_constant_columns", "drop_duplicate_rows"],  params_list=[{"n": 1000, "random_state": 0}, {"encode": ENCODE}, {}, {}], in_place=True)
    model_name = "catboost"
    dataset.batch_size = BS
    al_strategy = get_al_strategy(ALSTRATEGY)
    for metric in distance.AVAIL_DIST_METRICS:
    # for metric in ["cosine"]:
        fn = f"{DIST_CACHE}/{dsn}/{metric}_{ENCODE}.npy"
        cache_name = f"{dsn.replace('/', '_')}@{model_name}@{ALSTRATEGY}_{metric}_{ENCODE}@{SUFFIX}@x20.pkl"
        if ensure_path(os.path.join(LOCAL_DIR, "ppl_cache", cache_name), False):
            print(f"exp {cache_name} already exists, continue")
            continue
        if not ensure_path(fn, False):
            print(f"{metric} for {dsn} not found in {fn}, skipping ...")
            continue
        print(f"Using cache of {metric} for {dsn} with {ENCODE} encoding")
        setattr(al_strategy, "dist_cache_path", fn)
        ds = dataset.with_strategy(al_strategy, extra_params={"distance_metric": metric})
        ds.batch_size = BS
        make_ppl = get_pipeline(model_name)
        ppl = make_ppl(
            model=None,
            eval_metrics=get_metrics([]),
            # eval_metrics=get_metrics(["accuracy"]),
            seeds=[i for i in range(20)],
            n_times=20,
            dataset=ds,
            cat_idx=ds.info["cat_idx"],
            model_extra_params=model_extra_params,
            early_stop=TOTAL / BS
        )
        ppl.start()
        with open(os.path.join(LOCAL_DIR, "ppl_cache_pred", cache_name), "wb") as f:
            pickle.dump(ppl.stats, f)
    