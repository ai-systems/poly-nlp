from typing import List

import ujson as json
from loguru import logger
from overrides import overrides
from prefect import Task
from tqdm import tqdm


class ARCDatasetExtraction(Task):
    """Defines the ARC dataset
    """

    @overrides
    def run(self, path: str, fold: str):
        """Loads data from file

        Args:
            path (str): path for ARC dataset

        Returns:
            List[ARC]: List of ARC model
        """
        arc_data_question = {}
        with open(path, "r") as f:
            for line in tqdm(f, "Extracting ARC Dataset"):
                data = json.loads(line)
                choices = {
                    choice["label"]: choice["text"]
                    for choice in data["question"]["choices"]
                }
                arc_data_question[data["id"]] = {
                    "id": data["id"],
                    "question": data["question"]["stem"],
                    "answer": choices[data["answerKey"]],
                    "fold": fold,
                    "choices": choices,
                    "answerKey": data["answerKey"],
                }
        return arc_data_question
