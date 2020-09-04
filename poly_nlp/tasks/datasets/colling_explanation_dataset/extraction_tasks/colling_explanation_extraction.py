import datetime
import glob
import re
from collections import defaultdict
from enum import Enum
from typing import Dict, List
from uuid import uuid4

import pandas as pd
from dynaconf import settings
from loguru import logger
from overrides import overrides
from prefect import Task
from tqdm import tqdm


class CollingExplanationExtractionTask(Task):
    @overrides
    def run(self, expl_bank_data_file: str,) -> Dict[str, Dict]:
        return self.process_sheet(expl_bank_data_file,)

    @staticmethod
    def process_sheet(question_explanation_path: str,) -> Dict[str, Dict]:
        logger.info(f"Extracting data from {question_explanation_path}")

        QID = "questionID"
        QUESTION = "question"
        ANSWER_KEY = "AnswerKey"
        JUSTIFICATION = "justification"

        expl_df = pd.read_csv(question_explanation_path, encoding="utf-8")
        expl_items = {}
        choices_re = [
            ("A", re.compile("(?<=\([A|1]\)).*(?=\([B|2]\))")),
            ("B", re.compile("(?<=\([B|2]\)).*(?=\([C|3]\))")),
            ("C", re.compile("(?<=\([C|3]\)).*(?=\([D|4]\))")),
            ("D", re.compile("(?<=\([D|4]\)).*")),
        ]
        question_re = re.compile("\([ABCD]\).*")
        question_num_re = re.compile("\([1234]\).*")
        for _, row in tqdm(expl_df.iterrows(), total=expl_df.shape[0]):
            if not pd.isna(row[QUESTION]):
                id = row[QID]
                question = row[QUESTION]

                if pd.isna(row[JUSTIFICATION]) or row[JUSTIFICATION] == "":
                    logger.warning(f"{id} have no justifications. Skipping it")
                    continue
                choices = {
                    choice_re[0]: choice_re[1].findall(question)[0].strip()
                    if len(choice_re[1].findall(question)) > 0
                    else ""
                    for choice_re in choices_re
                }
                if row[ANSWER_KEY] in ["1", "2", "3", "4"]:
                    answerkey = chr((int(row[ANSWER_KEY]) - 1) + ord("A"))
                else:
                    answerkey = row[ANSWER_KEY]
                # List of explanation ids
                answer = choices[answerkey] if answerkey in choices else ""

                question_explanation = {
                    "id": id,
                    "question": question_num_re.sub(
                        "", question_re.sub("", question)
                    ).strip(),
                    "answer": answer,
                    "fold": "Easy",
                    # colling_expalanation=colling_expalanation,
                    # knowledge_type=knowledge_type,
                    # school_grade=school_grade,
                    "choices": choices,
                    "answerKey": answerkey,
                }
                expl_items[id] = question_explanation
        return expl_items
