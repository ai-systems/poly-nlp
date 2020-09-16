import math
import multiprocessing
from functools import reduce
from typing import Dict

import pandas as pd
import ray
from loguru import logger
from overrides import overrides
from prefect import Task
from tqdm import tqdm

from poly_nlp.utils.data_manipulation.data_manipulation import create_dict_chunks

UNCERTAINITY_PRIORTY = {
    "L1": 0,
    "L2": 1,
    "Uncertain": 3,
    "L3": 4,
    None: 6,
}

DICT_MAPPING = {"Certain": "L3", "L-inv": "L2"}


class UncertainityExtractionTask(Task):
    @ray.remote
    def process_data(pos: int, dict_mappings: Dict, sentences: Dict):
        results = {}
        for id, sentence in tqdm(sentences.items(), position=pos, mininterval=1):
            results[id] = None
            for word, label in dict_mappings.items():
                if (
                    word in sentence.lower()
                    and UNCERTAINITY_PRIORTY[results[id]] > UNCERTAINITY_PRIORTY[label]
                ):
                    results[id] = label
        return results

    @overrides
    def run(self, gen_dict_path: str, bio_dict_path: str, sentences: Dict[str, str]):
        gen_dict = {
            row[0]: DICT_MAPPING.get(row[1], row[1])
            for _, row in pd.read_csv(gen_dict_path, sep="\t", header=1).iterrows()
        }
        bio_dict = {
            row[0]: DICT_MAPPING.get(row[1], row[1])
            for _, row in pd.read_csv(bio_dict_path, sep="\t", header=1).iterrows()
        }

        dict_mappings = {**bio_dict, **gen_dict}

        batch_count = multiprocessing.cpu_count()
        batch_size = math.ceil(len(sentences) / batch_count)

        logger.info(f"Batch_size = {batch_size}")
        logger.info(f"Number of batches = {batch_count}")

        batches = create_dict_chunks(sentences, batch_size)

        batch_results = ray.get(
            [
                self.process_data.remote(pos, dict_mappings, batch)
                for pos, batch in enumerate(batches)
            ]
        )
        results = reduce(lambda x, y: {**x, **y}, batch_results, {})
        return results
