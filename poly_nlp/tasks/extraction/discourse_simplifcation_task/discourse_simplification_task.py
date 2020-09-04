import math
import multiprocessing
import os
from functools import reduce
from typing import Dict, List
from uuid import uuid4

import ray
import ujson as json
from loguru import logger
from prefect import Task
from tqdm import tqdm

from covid_fact_bank.utils.data_manipulation.data_manipulation import create_dict_chunks
from covid_fact_bank.utils.print_utils import suppress_stdout
from covid_fact_bank.utils.service import ServiceCaller

LEXICAL_CLUES = ["This", "That"]


class DiscourseSimplificationTask(Task):
    @ray.remote
    def process_data(pos, model_path, sentences):
        os.environ["CLASSPATH"] = model_path
        import jnius_config

        jnius_config.add_options("-Xmx4056M")
        from jnius import autoclass

        enum = autoclass(
            "org.lambda3.text.simplification.discourse.processing.ProcessingType"
        )
        model = autoclass(
            "org.lambda3.text.simplification.discourse.processing.DiscourseSimplifier"
        )()

        results = {}
        for id, sentence in tqdm(sentences.items(), position=pos, mininterval=1):
            try:
                with suppress_stdout():
                    result = model.doDiscourseSimplification(
                        str(sentence), enum.SEPARATE
                    ).serializeToJSON()
                    o_id = ray.put(json.decode(result))
                    results[id] = o_id
            except:

                logger.error(f"Unable to parse {sentence}")
                results[id] = None
        return results

    @staticmethod
    def build_representation(result: Dict, original_sentence) -> Dict:
        if result == None:
            s_id = str(uuid4())
            return {
                "extracted_sentences": {s_id: original_sentence},
                "core_facts": {s_id: {}},
            }
        extracted_elements = {
            s_id: element
            for sentence in result["sentences"]
            for s_id, element in sentence["elementMap"].items()
        }

        extracted_sentences = {}
        for element in extracted_elements.values():
            simple_contexts = []
            for context in element["simpleContexts"]:
                new_id = str(uuid4())
                simple_contexts.append(
                    {"targetID": new_id, "relation": context["relation"]}
                )
                extracted_sentences[new_id] = {
                    "text": context["text"],
                    "is_core": False,
                    "links": [],
                }
            element["linkedContexts"] = element["linkedContexts"] + simple_contexts

        linked_context_ids = set(
            [
                context["targetID"]
                for element in extracted_elements.values()
                for context in element["linkedContexts"]
            ]
        )

        extracted_sentences = {
            **extracted_sentences,
            **{
                id: {
                    "text": element["text"],
                    "is_core": not any(
                        [element["text"].startswith(clue) for clue in LEXICAL_CLUES]
                    )
                    and (
                        len(element["linkedContexts"]) > 0
                        or id not in linked_context_ids
                    ),
                    "links": element["linkedContexts"],
                }
                for id, element in extracted_elements.items()
            },
        }

        return extracted_sentences

    def run(self, sentences: Dict[str, str], model_path: str) -> Dict[str, List]:
        """

        Perform discourse simplification

        Arguments:
            sentences {Dict[str, str]} -- [dictionary of sentences]
            parameters {List[Dict]} -- {port, host and workers}

        Returns:
            Dict[str, List] -- service result
        """

        logger.info(f"Loading model from {model_path}")
        if not os.path.isfile(model_path):
            logger.error(f"Discourse model jar not found in {model_path}")
            raise FileNotFoundError

        batch_count = multiprocessing.cpu_count()
        batch_size = math.ceil(len(sentences) / batch_count)

        logger.info(f"Batch_size = {batch_size}")
        logger.info(f"Number of batches = {batch_count}")

        batches = create_dict_chunks(sentences, batch_size)

        logger.info("Running service.")

        batch_results = ray.get(
            [
                self.process_data.remote(pos, model_path, batch)
                for pos, batch in enumerate(batches)
            ]
        )
        results = {
            id: ray.get(o_id)
            for id, o_id in reduce(lambda x, y: {**x, **y}, batch_results, {}).items()
        }

        simplification_output = {
            id: self.build_representation(result, sentences[id])
            for id, result in results.items()
        }
        return simplification_output
