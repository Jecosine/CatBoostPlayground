import os.path
from abc import ABC, abstractmethod


class BasePlot(ABC):
    """
    Base class for plotting
    """
    _name: str

    def __init__(self, output_path: str = "./plots", new_dir=True):
        self.output_path = output_path if not new_dir else os.path.join(output_path, self._name)

    @abstractmethod
    def preprocess(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def plot(self, *args, **kwargs):
        raise NotImplementedError
