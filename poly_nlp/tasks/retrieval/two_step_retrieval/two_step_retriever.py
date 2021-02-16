import string
from heapq import nlargest

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Q, Search
from elasticsearch_dsl.query import Match
from loguru import logger
from nltk.corpus import stopwords
from overrides import overrides
from poly_nlp.parallel.ray_executor import RayExecutor
from poly_nlp.tasks.extraction.entity_extraction import EntityExtractionTask
from poly_nlp.tasks.extraction.spacy_processor_task import SpacyProcessorTask
from poly_nlp.tasks.retrieval.bm25 import BM25SearchTask
from prefect import Task
from tqdm import tqdm


class TwoStepRetiver(Task):
    """Implementation of the the Two step retriver from the QASC dataset paper"""

    @staticmethod
    def query(pos, input, query_index, k, l, m, corpus):
        first_hop_retrieval = {}
        second_hop_retrieval = {}
        retrieval = {}
        client = Elasticsearch()
        client.indices.clear_cache()

        for id, val in tqdm(input.items()):
            s = (
                Search(using=client, index=query_index)
                .query("match", fact=val)
                .params(request_timeout=30)
            )
            response = s[:k].execute()

            s = Search(using=client, index=query_index)
            first_hop_retrieval[id] = {}
            second_hop_retrieval[id] = {}

            for hit in response:
                first_hop_retrieval[id][hit.meta.id] = hit.meta.score
            for f_id in first_hop_retrieval[id]:

                lemmatized_q = set(
                    val.lower()
                    .translate(str.maketrans("", "", string.punctuation))
                    .split()
                )
                lemmatized_fact = set(
                    corpus[f_id]["fact"]
                    .lower()
                    .translate(str.maketrans("", "", string.punctuation))
                    .split(" ")
                )

                fact_diff_q = " ".join(
                    [word for word in list(lemmatized_fact.difference(lemmatized_q))]
                )
                q_diff_fact = " ".join(
                    [word for word in list(lemmatized_q.difference(lemmatized_fact))]
                )

                s.query = Q(
                    "bool",
                    must=[Q("match", fact=q_diff_fact), Q("match", fact=fact_diff_q)],
                )
                try:
                    response = s[:l].execute()

                    for hit in response:
                        if f_id != hit.meta.id:
                            if hit.meta.id in second_hop_retrieval[id]:
                                second_hop_retrieval[id][hit.meta.id] = max(
                                    second_hop_retrieval[id][hit.meta.id],
                                    hit.meta.score,
                                )
                            else:
                                second_hop_retrieval[id][hit.meta.id] = hit.meta.score
                except:
                    logger.error(f"{q_diff_fact} and {fact_diff_q} data too large")

            # retrieval[id] = {**first_hop_retrieval[id], **second_hop_retrieval[id]}
            retrieval[id] = {
                **first_hop_retrieval[id],
                **{
                    f_id: second_hop_retrieval[id][f_id]
                    for f_id in nlargest(
                        m - k,
                        second_hop_retrieval[id],
                        key=second_hop_retrieval[id].get,
                    )
                },
            }
            client.indices.clear_cache()
        return retrieval

    @overrides
    def run(self, query, corpus, query_index, k=30, l=5, m=100):
        logger.info(f"Running query index on {query_index}")

        results = RayExecutor().run(
            query,
            self.query,
            {"query_index": query_index, "k": k, "l": l, "m": m, "corpus": corpus},
            batch_count=8,
        )
        # results = self.query(
        #     pos=0,
        #     input=query,
        #     **{"query_index": query_index, "k": k, "l": l, "m": m, "corpus": corpus},
        # )
        return results
