import math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from overrides import overrides

from .metric import METRIC_LABEL, MODEL_NAME, SCORE, SPLIT_NAME, Metric


class MAP(Metric):
    def __init__(
        self,
        actual: Dict[str, List],
        predicted: [str, List],
        model_split: Tuple[str, str] = (None, None),
        k=None,
        metric_label="MAP",
    ):
        self.actual = {key: value for key, value in actual.items() if len(value) > 0}
        self.predicted = predicted
        self.k = k
        self.model_split = model_split
        self.metric_label = metric_label

    @staticmethod
    def compute_ranks(true, pred):
        ranks = []

        if not true or not pred:
            return ranks

        targets = list(true)

        # I do not understand the corresponding block of the original Scala code.
        for i, pred_id in enumerate(pred):
            for true_id in targets:
                if pred_id == true_id:
                    ranks.append(i + 1)
                    targets.remove(pred_id)
                    break

        # Example: Mercury_SC_416133
        if targets:
            for _ in targets:
                ranks.append(10 ** 9)

        return ranks

    @staticmethod
    def average_precision(actual, predicted):
        total = 0.0
        ranks = MAP.compute_ranks(actual, predicted)
        if not ranks:
            return total

        for i, rank in enumerate(ranks):
            precision = float(i + 1) / float(rank) if rank > 0 else math.inf
            total += precision

        return total / len(ranks)

    def calc_metric(self):
        predicted = defaultdict(lambda: [], self.predicted)
        return round(
            np.mean(
                [
                    self.average_precision(a, predicted[id])
                    for id, a in self.actual.items()
                ]
            ),
            4,
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
