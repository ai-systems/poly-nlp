import glob
import os
from functools import reduce
from typing import Dict, List
from uuid import uuid4

import pandas as pd
from dynaconf import settings
from loguru import logger
from overrides import overrides
from prefect import Task
from tqdm import tqdm

from .worldtree import WorldTreeVersion

UID = "[SKIP] UID"


class TableStoreExtractionTask(Task):
    @overrides
    def run(
        self,
        table_store_path: str,
        original_map=True,
        table_categories_path=None,
        worldtree_version=WorldTreeVersion.WorldTree_V2,
    ) -> Dict[str, Dict]:
        table_categories = {}
        if table_categories_path is not None:
            logger.info("Processing table categories file")
            with open(table_categories_path) as f:
                for line in f:
                    cat, k_type = line.split()
                    k_type = k_type.split("/")[0]  # For RET/LEX
                    if worldtree_version == WorldTreeVersion.WorldTree_V2:
                        table_categories[cat.replace("PROTO-", "")] = k_type.strip()
                    else:
                        table_categories[cat] = k_type.strip()
        logger.info(f"Extracting table store from: {table_store_path}")
        # Get all tsv files
        table_store_files = glob.glob(f"{table_store_path}/*.tsv")
        table_store = reduce(
            lambda store, file_name: {
                **store,
                **self.process_store(
                    file_name,
                    os.path.basename(file_name),
                    original_map,
                    table_categories,
                ),
            },
            tqdm(table_store_files),
            {},
        )
        return table_store

    @staticmethod
    def process_store(
        file_name: str, table_name: str, original_map=True, table_categories={}
    ) -> Dict[str, Dict]:
        """Process individual file stores
        """
        table_df = pd.read_csv(file_name, sep="\t")
        split_table_items = {}
        orig_table_items = {}
        try:
            for _, row in tqdm(table_df.iterrows(), total=table_df.shape[0]):
                id = row[UID]
                explanation = reduce(
                    lambda expl, items: {**expl, items[0]: None}
                    if pd.isna(items[1])
                    else {**expl, items[0]: items[1]},
                    row.items(),
                    {},
                )
                filtered_explanation = {
                    k: v for k, v in explanation.items() if k != UID and "SKIP" not in k
                }
                sentence_explanation = [
                    str(item[1])
                    for item in list(
                        filter(
                            lambda item: not "SKIP" in item[0]
                            and not item[1] is None
                            and not pd.isna(item[1]),
                            explanation.items(),
                        )
                    )
                ]
                # Split the fact based on semicolon
                count = 0
                tot = 0
                indexes = []
                for token in sentence_explanation:
                    if ";" in token:
                        tot += 1
                        indexes.append(count)
                    count += 1
                new_facts = []
                if tot == 0:
                    new_facts = [" ".join(sentence_explanation).replace("  ", " ")]
                else:
                    stack = [sentence_explanation]
                    for index in indexes:
                        tokens = sentence_explanation[index].split(";")
                        new_stack = []
                        for f in stack:
                            for token in tokens:
                                new_stack.append(f[:index] + [token] + f[index + 1 :])
                        stack = new_stack
                    for f in stack:
                        new_facts.append(" ".join(f).replace("  ", " "))
                for fact in new_facts:
                    split_table_items[str(uuid4())] = {
                        "id": id,
                        "explanation": explanation,
                        "sentence_explanation": sentence_explanation,
                        "table_name": table_name.split(".")[0],
                        "knowledge_category": table_categories.get(
                            table_name.split(".")[0], "RET"
                        ),
                        "filtered_explanation": filtered_explanation,
                        "fact": fact,
                    }
                orig_table_items[id] = {
                    "id": id,
                    "explanation": explanation,
                    "sentence_explanation": sentence_explanation,
                    "table_name": table_name.split(".")[0],
                    "knowledge_category": table_categories.get(
                        table_name.split(".")[0], "RET"
                    ),
                    "filtered_explanation": filtered_explanation,
                }
        except KeyError:
            logger.warning(f"Unable to process {table_name} because of KeyError")
        if original_map:
            return orig_table_items
        else:
            return split_table_items
