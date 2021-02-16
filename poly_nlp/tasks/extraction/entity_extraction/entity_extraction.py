import math
import multiprocessing
import time
from functools import reduce

import ray
from loguru import logger
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.util import ngrams
from overrides import overrides
from poly_nlp.utils.data_manipulation.data_manipulation import create_dict_chunks
from prefect import Task
from tqdm import tqdm

stop_english = stopwords.words("english")


class EntityExtractionTask(Task):
    @staticmethod
    def entity_extraction(text, lemmas_in_words):
        entites = set()
        for count in range(2, 0, -1):
            tokens = text.split()
            for ngram in list(ngrams(tokens, count)):
                if "_".join(ngram) in lemmas_in_words:
                    # text = text.replace(" ".join(ngram), "")
                    entites.add(" ".join(ngram))
        return entites

    @staticmethod
    def recognize_entities(string_or_list):
        if type(string_or_list) == list:
            text_value = " ".join(string_or_list)
        else:
            text_value = string_or_list

        entities = []
        temp = []
        for word in word_tokenize(text_value):
            if not word.lower() in stop_english:
                temp.append(word.lower())
        tokenized_string = word_tokenize(" ".join(temp))
        head_index = 0
        word_index = 0
        for word in tokenized_string:
            check_index = len(tokenized_string)
            final_entity = ""
            if word_index > head_index:
                head_index = word_index
            while check_index > head_index:
                if (
                    len(wn.synsets("_".join(tokenized_string[head_index:check_index])))
                    > 0
                ):
                    final_entity = " ".join(tokenized_string[head_index:check_index])
                    entities.append(final_entity)
                    break
                check_index -= 1
            head_index = check_index
            word_index += 1
        return entities

    @ray.remote
    def process_batch(text_input):
        extracted_entites = {
            id: set(EntityExtractionTask.recognize_entities(text))
            # id: text.split()
            for id, text in text_input.items()
        }
        return extracted_entites

    @overrides
    def run(self, text_input, is_fact=False):
        batch_count = multiprocessing.cpu_count()
        batch_size = math.ceil(len(text_input) / batch_count)

        logger.info(f"Batch_size = {batch_size}")
        logger.info(f"Number of batches = {batch_count}")

        batches = create_dict_chunks(text_input, batch_size)

        logger.info("Entity Extraction")
        start = time.time()

        batch_results = ray.get(
            [
                self.process_batch.remote(text_input=batch)
                for pos, batch in enumerate(tqdm(batches))
            ]
        )

        extracted_entites = reduce(lambda x, y: {**x, **y}, batch_results, {})
        end = time.time()
        logger.success(f"Enitity Extraction successful. Time taken: {end-start}")
        return extracted_entites
