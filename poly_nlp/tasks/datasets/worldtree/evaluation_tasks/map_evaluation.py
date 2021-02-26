from typing import Dict

from loguru import logger
from overrides import overrides
from poly_nlp.evaluate.metrics import MAP, RecallTopK
from prefect import Task


class WorldTreeMAPEvaluationTask(Task):
    @overrides
    def run(self, prediction: Dict, gold_explanations: Dict, log=True):
        map_overall = MAP(gold_explanations, prediction).calc_metric()
        ###recall_topk = RecallTopK(gold_explanations, prediction).calc_metric()
        if log:
            logger.success(f"MAP Overall {map_overall}")
            # logger.success(f"Recall Overall {recall_topk}")
        return map_overall
