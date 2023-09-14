from typing import List

import matplotlib.pyplot as plt
from ..base import BasePlot
from ...datasets import Dataset


class PLALSelection(BasePlot):
    _name = "pl_selection"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, *args, **kwargs):
        pass

    def plot(self, ppl_datas: List[dict], dss: List[Dataset], *args, **kwargs):
        # two type of plots:
        # 1. plot for each run, showing the selection by colors 2d only
        # 2. plots for each selection
        # stats_len = kwargs.get("stats_len", 10)
        # plot type 1
        cmap_name = kwargs.get("cmap_name", "Oranges")
        for stats, ds in zip(ppl_datas, dss):
            x = ds.u_x.copy()
            snapshots = stats["snapshot"]
            cmap = plt.get_cmap(cmap_name, 10)
            x1, x2 = x.iloc[:, 0], x.iloc[:, 1]
            pre_idx = None
            for idx, snapshot in enumerate(snapshots):
                fig, ax = plt.subplots(figsize=(5, 5))
                if pre_idx is not None:
                    ax.scatter(x1.loc[pre_idx], x2.loc[pre_idx], cmap=cmap, c=2)
                ax.scatter(x1.loc[snapshot], x2.loc[snapshot], cmap=cmap, c=9)



    def automation(self):
        dss = ["iris", "adult"]
        alms = ["random", "uncertain", "ranked_batch"]

