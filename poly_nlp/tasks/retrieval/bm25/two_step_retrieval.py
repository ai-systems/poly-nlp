from typing import Dict

from loguru import logger
from nltk.corpus import stopwords
from overrides import overrides
from poly_nlp.parallel.ray_executor import RayExecutor
from poly_nlp.tasks.extraction.spacy_processor_task import SpacyProcessorTask
from prefect import Flow, Task
from tqdm import tqdm

from .bm25_task import BM25FitTask, BM25SearchTask


class TwoStepRetreival(Task):
    @staticmethod
    def bm25_search(lemmatized_query, corpus, limit):
        search_task = BM25SearchTask()
        build_index_task = BM25FitTask()

        with Flow("First step retrieval") as flow:
            retriever = build_index_task(corpus)
            results = search_task(lemmatized_query, retriever, limit=limit)

        state = flow.run()
        return state.result[results]._result.value

    @staticmethod
    def prepare_data(pos, input, corpus, query):
        processed_query = {}
        for q_id, results in tqdm(input.items()):
            for r_id in results:
                processed_query[f"{q_id}|{r_id}"] = f"{query[q_id]} {corpus[r_id]}"

        return processed_query

    @overrides
    def run(self, query: Dict[str, str], corpus: Dict[str, str], k=70):
        logger.info("Running First Step Retrieval")

        stop = stopwords.words("english")

        lemmatized_query = SpacyProcessorTask().run(
            query_dict=query,
            processor=lambda word: word.lemma_,
            filter_fn=lambda word: word.lemma_ not in stop,
        )

        first_step_results = self.bm25_search(
            lemmatized_query=lemmatized_query, corpus=corpus, limit=k
        )

        logger.info("Preparing data for second step retrieval")

        prepared_data = RayExecutor().run(
            first_step_results, self.prepare_data, {"corpus": corpus, "query": query}
        )

        lemmatized_second_query = SpacyProcessorTask().run(
            query_dict=prepared_data,
            processor=lambda word: word.lemma_,
            filter_fn=lambda word: word.lemma_ not in stop,
        )

        logger.info("Running Second Step Retrieval")

        second_step_results = self.bm25_search(
            lemmatized_query=lemmatized_second_query, corpus=corpus, limit=k
        )

        return {
            "results": {1: first_step_results, 2: second_step_results},
            "lemmatized": {1: lemmatized_query, 2: lemmatized_second_query},
        }
