from abc import ABC, abstractmethod

SCORE = "score"
METRIC_LABEL = "metric_label"
MODEL_NAME = "model_name"
SPLIT_NAME = "split_name"


class Metric(ABC):
    @abstractmethod
    def calc_metric(self, actual, metric, **kwargs):
        ...

    @abstractmethod
    def __call__(self):
        ...

