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


class WorldTreeVersion(Enum):
    WorldTree_V1 = "worldtree_v1"
    WorldTree_V2 = "worldtree_v2"


class WorldTreeExtractionTask(Task):
    @overrides
    def run(
        self,
        expl_bank_data_file: str,
        worldtree_version: WorldTreeVersion,
        skip_missing_explanations=False,
        skip_empty_explanations=True,
    ) -> Dict[str, Dict]:
        return self.process_sheet(
            expl_bank_data_file,
            worldtree_version,
            skip_missing_explanations,
            skip_empty_explanations,
        )

    @staticmethod
    def process_sheet(
        question_explanation_path: str,
        worldtree_version: WorldTreeVersion,
        skip_missing_explanations: bool,
        skip_empty_explanations,
    ) -> Dict[str, Dict]:
        logger.info(
            f"Extracting data from {question_explanation_path}, WorldTree Version {worldtree_version}"
        )
        if worldtree_version == WorldTreeVersion.WorldTree_V1:
            QID = "questionID"
            EXPLANATION = "explanation"
            FOLD = "fold"
            QUESTION = "Question"
            ANSWER_KEY = "AnswerKey"
            FLAG = "flags"
            EXAM_NAME = "examName"
            GRADE = "grade"
        else:
            QID = "QuestionID"
            EXPLANATION = "explanation"
            FOLD = "FocusNotes"
            QUESTION = "question"
            ANSWER_KEY = "AnswerKey"
            FLAG = "flags"
            EXAM_NAME = "examName"
            GRADE = "grade"

        expl_df = pd.read_csv(question_explanation_path, sep="\t", encoding="utf-8")
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
            # Check for nan values.
            # WARNING: Sometimes explanation is empty
            if (
                not skip_missing_explanations
                and skip_empty_explanations
                and "EMPTY" in str(row[FLAG])
            ):
                logger.warning(f"{row[QID]} flag is not sucess. Skipping")
                continue
            if pd.isna(row[EXPLANATION]) and skip_missing_explanations:
                logger.warning(f"{row[QID]} does not have any explanations")
                continue
            if not pd.isna(row[QUESTION]):
                id = row[QID]
                question = row[QUESTION]
                fold = row[FOLD]
                topic = (
                    None
                    if worldtree_version == WorldTreeVersion.WorldTree_V1
                    else row["topic"].split(",")
                )
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
                if not pd.isna(row[EXPLANATION]):
                    explanation = {
                        expl.split("|")[0]: expl.split("|")[1]
                        for expl in row[EXPLANATION].split(" ")
                    }
                else:
                    explanation = {}
                answer = choices[answerkey] if answerkey in choices else ""
                # Filter to those only exisiting in table stores

                # colling_expalanation = row[COLLING_EXPLANATION].split(
                # ' ') if not pd.isna(row[COLLING_EXPLANATION]) else []
                # knowledge_type = row[KNOWLEDGE_TYPE]
                # school_grade = row[SCHOOL_GRADE]
                question_explanation = {
                    "id": id,
                    "question": question_num_re.sub(
                        "", question_re.sub("", question)
                    ).strip(),
                    "explanation": explanation,
                    "answer": answer,
                    "fold": fold,
                    "topic": topic,
                    "grade": row[GRADE],
                    "examName": row[EXAM_NAME],
                    # colling_expalanation=colling_expalanation,
                    # knowledge_type=knowledge_type,
                    # school_grade=school_grade,
                    "choices": choices,
                    "answerKey": answerkey,
                }
                expl_items[id] = question_explanation
        return expl_items
