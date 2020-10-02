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
from prefect import Task
from tqdm import tqdm

from poly_nlp.utils.data_manipulation.data_manipulation import create_dict_chunks

stop = stopwords.words("english")


class SpacyProcessorTask(Task):
    @staticmethod
    def tokenize(query_dict, lemmas, stop):
        nlp = spacy.load("en_core_web_sm")
        query_output = {}
        for id, query in query_dict.items():
            query_output[id] = [word for word in nlp(query)]
        return query_output

    @overrides
    def run(self, query_dict: Dict, lematizer_path: str, no_parallel=False):
        ray_executor = RayExecutor()
        query_output = ray_executor.run(
            query_dict,
            self.tokenize,
            {},
        )

        return query_output
