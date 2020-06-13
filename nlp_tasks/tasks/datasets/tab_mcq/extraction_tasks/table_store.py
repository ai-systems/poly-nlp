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


class AristoTableStoreExtractionTask(Task):
    @overrides
    def run(self, table_store_path: str, table_name) -> Dict[str, Dict]:
        table_categories = {}
        logger.info(f"Extracting table store from: {table_store_path}")
        # Get all tsv files
        table_store_files = glob.glob(f"{table_store_path}/*.tsv")
        table_store = reduce(
            lambda store, file_name: {
                **store,
                **self.process_store(file_name, table_name),
            },
            tqdm(table_store_files),
            {},
        )
        return table_store

    @staticmethod
    def process_store(file_name: str, table_name: str) -> Dict[str, Dict]:
        """Process individual file stores
        """
        table_df = pd.read_csv(file_name, sep="\t")
        table_items = {}
        try:
            for _, row in tqdm(table_df.iterrows(), total=table_df.shape[0]):
                id = str(uuid4())
                table_items[id] = {
                    "id": id,
                    "fact": " ".join([str(val) for val in row.values]),
                    "table_name": table_name,
                }
        except KeyError:
            logger.warning(f"Unable to process {table_name} because of KeyError")
        return table_items
