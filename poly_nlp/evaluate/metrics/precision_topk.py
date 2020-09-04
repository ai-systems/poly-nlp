from typing import Dict, List, Tuple

import numpy as np
from overrides import overrides

from .metric import METRIC_LABEL, MODEL_NAME, SCORE, SPLIT_NAME, Metric


class PrecisionTopK(Metric):
    def __init__(
        self,
        actual: Dict[str, List],
        predicted: [str, List],
        k=None,
        model_split: Tuple[str, str] = (None, None),
        metric_label="Precision TopK",
    ):
        self.actual = actual
        self.predicted = predicted
        self.k = k
        self.model_split = model_split
        self.metric_label = metric_label

    @staticmethod
    def avg_precision(actual, predicted, k=None):
        if k is None:
            k = len(predicted)
        if len(predicted) > k:
            predicted = predicted[:k]

        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0

        if not actual:
            return 1.0

        return num_hits / k

    def calc_metric(self):
        return round(
            np.mean(
                [
                    self.avg_precision(a, self.predicted[id], self.k)
                    for id, a in self.actual.items()
                ]
            )
            * 100,
            2,
        )

    def __call__(self, round_val=4, invert=False):
        map_score = self.calc_metric()
        return {
            SCORE: round(map_score, round_val),
            METRIC_LABEL: self.metric_label,
            MODEL_NAME: self.model_split[0]
            if self.model_split[0] is not None
            else "DEFAULT",
            SPLIT_NAME: self.model_split[1]
            if self.model_split[1] is not None
            else "DEFAULT",
        }
