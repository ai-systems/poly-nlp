from typing import Dict

from loguru import logger
from overrides import overrides
from prefect import Task

from nlp_tasks.evaluate.metrics import MAP


class WorldTreeMAPEvaluationTask(Task):
    @overrides
    def run(self, prediction: Dict, gold_explanations: Dict):
        map_overall = MAP(gold_explanations, prediction).calc_metric()
        logger.success(f"MAP Overall {map_overall}")
        return map_overall
