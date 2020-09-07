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


class GenericsKBExtractionTask(Task):
    @overrides
    def run(
        self, table_store_path: str, filter_fn=None, process_fn=None
    ) -> Dict[str, Dict]:
        table_categories = {}
        logger.info(f"Extracting generics kb from: {table_store_path}")
        logger.info(f"Filter function provided: {filter_fn is not None}")
        # Get all tsv files
        table_store = self.process_store(
            table_store_path, filter_fn=filter_fn, process_fn=process_fn
        )
        logger.success(
            f"GenericsKB Extraction successful. Number of facts {len(table_store)}"
        )
        return table_store

    @staticmethod
    def process_store(
        file_name: str, filter_fn=None, process_fn=None
    ) -> Dict[str, Dict]:
        """Process individual file stores
        """
        table_df = pd.read_csv(file_name, sep="\t")
        table_items = {}
        for _, row in tqdm(table_df.iterrows(), total=table_df.shape[0]):
            id = str(uuid4())
            skip = False
            fact = {
                "id": id,
                "fact": row["GENERIC SENTENCE"]
                if process_fn is None
                else process_fn(row["GENERIC SENTENCE"]),
                "source": row["SOURCE"],
                "score": row["SCORE"],
            }
            if filter_fn is not None:
                skip = filter_fn(fact)
            if not skip:
                table_items[id] = fact
        return table_items
