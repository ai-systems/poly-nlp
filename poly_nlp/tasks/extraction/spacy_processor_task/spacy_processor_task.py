import math
import multiprocessing
import time
from functools import reduce
from typing import Dict

import nltk
import ray
import spacy
from loguru import logger
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from overrides import overrides
from poly_nlp.parallel.ray_executor import RayExecutor
from poly_nlp.utils.data_manipulation.data_manipulation import create_dict_chunks
from prefect import Task
from tqdm import tqdm

stop = stopwords.words("english")


class SpacyProcessorTask(Task):
    @staticmethod
    def tokenize(pos, input, processor, filter_fn):
        nlp = spacy.load("en_core_web_sm")
        query_output = {}
        for id, query in input.items():
            query_output[id] = [
                processor(word) if processor is not None else word
                for word in nlp(query)
                if filter_fn(word)
            ]
        return query_output

    @overrides
    def run(self, query_dict: Dict, processor=None, filter_fn=None, is_parallel=True):
        logger.info(f"Running Spacy Task. Processor: {processor is not None}")
        if filter_fn is None:
            filter_fn = lambda word: True
        ray_executor = RayExecutor()
        query_output = ray_executor.run(
            query_dict,
            self.tokenize,
            {"processor": processor, "filter_fn": filter_fn},
            is_parallel=is_parallel,
        )

        return query_output
