import os.path
from typing import List, Callable

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np

from .base import BasePlot
from ..typing import ArrayLike


class PLMetric(BasePlot):
    _name = "pl_metric"

    def __init__(self, ax, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supported_errors = {
            "minmax": self.__minmax_errbar,
            "quantile": self.__quantile_errbar,
            "std": self.__std_errbar,
        }
        self.ax = ax
        self.fig = ax.get_figure()

    def __minmax_errbar(self, metrics: ArrayLike):
        med = np.median(metrics, axis=0)
        mx = metrics.max(axis=0)
        mn = metrics.min(axis=0)
        return med, mx, mn

    def __quantile_errbar(self, metrics: ArrayLike):
        med = np.median(metrics, axis=0)
        q1 = np.quantile(metrics, 0.25, axis=0)
        q3 = np.quantile(metrics, 0.75, axis=0)
        return med, q1, q3

    def __std_errbar(self, metrics: ArrayLike):
        mean = np.mean(metrics, axis=0)
        std = np.std(metrics, axis=0) / np.sqrt(metrics.shape[0])
        return mean, mean+std, mean-std

    def preprocess(self, *args, **kwargs):
        pass

    def errorbar(
        self,
        type: str,
        metrics: ArrayLike,
    ):
        if type not in self.supported_errors:
            raise RuntimeError(f"Error type {type} is not supported")
        return self.supported_errors[type](metrics)

    def plot(
        self,
        metric_name: str,
        instances: ArrayLike,
        metrics_n_times: ArrayLike,
        strategies: List[str],
        pre_plot: Callable = None,
        post_plot: Callable = None,
        plot_name: str = "plot.png",
        cmap: plt.colormaps = colormaps.get_cmap("gist_rainbow"),
        dpi: int = 300,
        errorbar: str = "quantile",
        title: str = None,
        multi_x: bool = False,
        *args,
        **kwargs,
    ):
        """
        Args:
            dpi:
            cmap:
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
        title = title or f"{metric_name} on different strategies"
        n_times = len(metrics_n_times)
        # if kwargs.get("ax"):
        #     ax = kwargs.get("ax")
        #     fig = ax.get_figure()
        # else:
        #     fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
        #     ax.set_title(title)
        ax, fig = self.ax, self.fig
        if pre_plot:
            pre_plot(metric_name, instances, metrics_n_times, strategies, ax, fig)
        # cm = plt.get_cmap("gist_rainbow")
        cm = cmap
        num_colors = n_times
        ax.set_prop_cycle("color", [cm(1.0 * i / num_colors) for i in range(num_colors)])
        for idx, metrics in enumerate(metrics_n_times):
            # med = np.median(metrics, axis=0)
            # mx = metrics.max(axis=0)
            # mn = metrics.min(axis=0)
            # q1 = np.quantile(metrics, 0.25, axis=0)
            # q3 = np.quantile(metrics, 0.75, axis=0)
            x = instances[idx] if multi_x else instances
            med, mx, mn = self.errorbar(errorbar, metrics)
            ax.fill_between(x, mn, mx, alpha=0.1)
            ax.plot(x, med, label=strategies[idx], marker="o", markersize=3)
        # ax.set_ylim(0, 1)
        ax.set_xlabel("Instances")
        ax.set_ylabel(f"{metric_name}")
        ax.legend()
        if post_plot:
            post_plot(metric_name, instances, metrics_n_times, strategies, ax, fig)
        

    def savefig(self, name: str):
        self.fig.savefig(os.path.join(self.output_path, name))
        plt.close(self.fig)