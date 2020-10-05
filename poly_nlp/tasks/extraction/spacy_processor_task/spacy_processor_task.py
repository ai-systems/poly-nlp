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
from tqdm import tqdcwin

stop = stopwords.words("english")


class SpacyProcessorTask(Task):
    @staticmethod
    def tokenize(pos, input):
        nlp = spacy.load("en_core_web_sm")
        query_output = {}
        for id, query in input.items():
            query_output[id] = [word for word in nlp(query)]
        return query_output

    @overrides
    def run(self, query_dict: Dict, no_parallel=False):
        ray_executor = RayExecutor()
        query_output = ray_executor.run(query_dict, self.tokenize, {},)

        return query_output
