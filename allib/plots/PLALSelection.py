import os.path
from typing import List, Callable

import matplotlib.pyplot as plt
from .base import BasePlot
from ..datasets import Dataset


class PLALSelection(BasePlot):
    _name = "pl_selection"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, *args, **kwargs):
        pass

    def plot(self, stats: dict, ds: Dataset, pre_plot: Callable, post_plot: Callable, *args, **kwargs):
        # two type of plots:
        # 1. plot for each run, showing the selection by colors 2d only
        # 2. plots for each selection
        # stats_len = kwargs.get("stats_len", 10)
        # plot type 1
        ds.reset()
        # data, label = ds._data, ds._label
        snapshots = stats["snapshot"]
        x1,  x2 = ds.u_x.iloc[:, 0], ds.u_x.iloc[:, 1]
        l = snapshots[0]
        for idx, snapshot in enumerate(snapshots):
            if idx == 0:
                continue
            fig, ax = plt.subplots(figsize=(5, 5))
            pre_plot(fig, ax, stats, ds, idx)
            # assert l is index type
            l = l.union(snapshot)
            ax.scatter(x1.drop(l), x2.drop(l), c="b", alpha=0.1, label="unlabeled")
            ax.scatter(x1[snapshot], x2[snapshot], c="r", maker="x", label="selection")
            ax.set_title(f"selection at iteration {idx}(start from 0)")
            ax.legend()
            post_plot(fig, ax, stats, ds, idx)
            fig.savefig(os.path.join(self.output_path, f"{idx}.png"))
