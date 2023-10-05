import os.path
from typing import List, Callable

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np

from .base import BasePlot
from ..typing import ArrayLike


class PLMetric(BasePlot):
    _name = "pl_metric"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, *args, **kwargs):
        pass

    def plot(
        self,
        metric_name: str,
        instances: ArrayLike,
        metrics_n_times: ArrayLike,
        strategies: List[str],
        pre_plot: Callable = None,
        post_plot: Callable = None,
        plot_name: str = "plot.png",
        *args,
        **kwargs,
    ):
        """
        Args:
            plot_name:
            metric_name:
            instances:
            metrics_n_times: of shape (n_strategies, n_times, n_iterations)
            strategies:
            pre_plot:
            post_plot:
            *args:
            **kwargs:

        Returns:

        """
        title = f"{metric_name} on different strategies"
        n_times = len(metrics_n_times)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(title)

        if pre_plot:
            pre_plot(metric_name, instances, metrics_n_times, strategies)
        cm = plt.get_cmap("gist_rainbow")
        num_colors = metrics_n_times.shape[0]
        ax.set_prop_cycle("color", [cm(1.0 * i / num_colors) for i in range(num_colors)])
        for idx, metrics in enumerate(metrics_n_times):
            med = np.median(metrics, axis=0)
            mx = metrics.max(axis=0)
            mn = metrics.min(axis=0)
            q1 = np.quantile(metrics, 0.25, axis=0)
            q3 = np.quantile(metrics, 0.75, axis=0)
            ax.fill_between(instances, q1, q3, alpha=0.1)
            ax.plot(instances, med, label=strategies[idx])
        # ax.set_ylim(0, 1)
        ax.set_xlabel("Instances")
        ax.set_ylabel(f"{metric_name}")
        ax.legend()
        if post_plot:
            post_plot(metric_name, instances, metrics_n_times, strategies)
        fig.savefig(os.path.join(self.output_path, plot_name))
        plt.close(fig)
