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
    def run(self, table_store_path: str) -> Dict[str, Dict]:
        table_categories = {}
        logger.info(f"Extracting generics kb from: {table_store_path}")
        # Get all tsv files
        table_store = self.process_store(table_store_path)
        return table_store

    @staticmethod
    def process_store(file_name: str) -> Dict[str, Dict]:
        """Process individual file stores
        """
        table_df = pd.read_csv(file_name, sep="\t")
        table_items = {}
        for _, row in tqdm(table_df.iterrows(), total=table_df.shape[0]):
            id = str(uuid4())
            table_items[id] = {
                "id": id,
                "fact": row["GENERIC SENTENCE"],
                "source": row["SOURCE"],
            }
        return table_items
