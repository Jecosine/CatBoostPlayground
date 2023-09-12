import matplotlib.pyplot as plt
from ..base import BasePlot


class PLALSelection(BasePlot):
    _name = "pl_selection"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, *args, **kwargs):
        pass

    def plot(self, ppl_datas: list, dss: list, *args, **kwargs):
        # two type of plots:
        # 1. plot for each run, showing the selection by colors
        # 2. plots for each selection
        # stats_len = kwargs.get("stats_len", 10)
        # plot type 1
        for stats, ds in zip(ppl_datas, dss):
            fig, ax = plt.subplots(figsize=(5, 5))


        pass


    def automation(self):
        dss = ["iris", "adult"]
        alms = ["random", "uncertain", "ranked_batch"]


