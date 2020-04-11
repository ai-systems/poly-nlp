import math
from collections import defaultdict
from functools import partial, reduce
from typing import Dict

import numpy as np
from loguru import logger
from overrides import overrides
from prefect import Task

from regra.data import Instance
from regra.tasks.datasets.explanation_bank.data_process_tasks.eb_bert_processing import (
    EBBankDataset,
)

FOLD = "fold"
CHALLENGE = "Challenge"
EASY = "Easy"


class QAEvaluationTask(Task):
    @staticmethod
    def evaluate_acc(
        dataset: EBBankDataset,
        eb_bank: Dict[str, Instance],
        indexes: np.array,
        preds: np.array,
        **kwargs,
    ):
        preds = preds[:, 1]
        q_ids = list(map(lambda index: dataset.get_id(index), indexes))
        scores, choices = defaultdict(lambda: -math.inf), defaultdict(lambda: None)
        for index, (id, choice) in enumerate(q_ids):
            if preds[index] > scores[id]:
                scores[id] = preds[index]
                choices[id] = choice
        total, correct = 0, 0
        t_easy, c_easy = 0, 0
        t_challenge, c_challenge = 0, 0
        for id, q_exp in eb_bank.items():
            if q_exp["answerKey"] in q_exp["choices"]:
                total += 1
                if q_exp[FOLD] == CHALLENGE:
                    t_challenge += 1
                elif q_exp[FOLD] == EASY:
                    t_easy += 1
                if q_exp["answerKey"] == choices[id]:
                    if q_exp[FOLD] == CHALLENGE:
                        c_challenge += 1
                    elif q_exp[FOLD] == EASY:
                        c_easy += 1
                    correct += 1
        acc = correct / total
        e_acc = c_easy / t_easy
        c_acc = c_challenge / t_challenge
        logger.info(f"Easy acc: {e_acc}, Challenge acc: {c_acc}")
        return acc

    @overrides
    def run(self, datasets, eb_bank, mode="dev"):
        return partial(self.evaluate_acc, dataset=datasets[1], eb_bank=eb_bank)
