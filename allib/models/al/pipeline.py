from typing import List, Callable, Optional, Tuple

from ..core import BaseModel
from ...datasets import Dataset
from ...typing import ArrayLike


# todo: handle tailing instances (merged to former?)
class ActiveLearningPipeline:
    def __init__(
            self,
            model: BaseModel,
            dataset: Dataset,
            eval_metrics: List[Callable],
            eval_set: Tuple[ArrayLike, ArrayLike],
            batch_size_updater: Optional[Callable],
    ):
        self.batch_size_updater = batch_size_updater
        self.model = model
        if len(eval_metrics) == 0:
            raise RuntimeError("Require at least one valid evaluating metrics")
        self.__eval_metrics = eval_metrics
        self.__eval_set = eval_set
        self.stats = {mc.__name__: [] for mc in self.__eval_metrics}
        self.dataset = dataset

    def apply_metrics(self):
        for mc in self.__eval_metrics:
            self.stats[mc.__name__].append(mc(self.model, *self.__eval_set))

    def epoch(self, train_x: ArrayLike, train_y: ArrayLike):
        """ [OVERRIDE NEEDED] Run one epoch

        Args:
            train_x:
            train_y:

        Returns:

        """
        pass

    def run(self):
        pass
