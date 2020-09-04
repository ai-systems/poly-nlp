from typing import List

import ujson as json
from loguru import logger
from overrides import overrides
from prefect import Task
from tqdm import tqdm


class QASCExtractionTask(Task):
    """Defines the QASC dataset
    """

    @overrides
    def run(self, path: str, fold: str):
        """Loads data from file

        Args:
            path (str): path for QASC dataset
        """
        arc_data_question = {}
        with open(path, "r") as f:
            for line in tqdm(f, "Extracting ARC Dataset"):
                data = json.loads(line)
                choices = {
                    choice["label"]: choice["text"]
                    for choice in data["question"]["choices"]
                }
                choices_para = {
                    choice["label"]: choice["para"] if "para" in choices else None
                    for choice in data["question"]["choices"]
                }
                arc_data_question[data["id"]] = {
                    "id": data["id"],
                    "question": data["question"]["stem"],
                    "answer": choices[data["answerKey"]],
                    "fold": fold,
                    "choices": choices,
                    "answerKey": data["answerKey"],
                    "choices_para": choices_para,
                    "fact1": data["fact1"],
                    "fact2": data["fact2"],
                }
        return arc_data_question
