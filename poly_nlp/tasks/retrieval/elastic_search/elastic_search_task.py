import multiprocessing

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, parallel_bulk
from elasticsearch_dsl import Document, Index, Text, analyzer, connections, tokenizer
from loguru import logger
from overrides import overrides
from poly_nlp.parallel.ray_executor import RayExecutor
from prefect import Task
from tqdm import tqdm


class Fact(Document):
    fact = Text(
        analyzer=analyzer(
            "fact_analyzer",
            tokenizer=tokenizer("standard"),
            filter=["stop", "lowercase"],
        )
    )

    class Index:
        name = "fact_corpus"
        settings = {
            "number_of_shards": 100,
        }


class BuildElasticIndex(Task):
    def run(self, corpus, index_name="fact_corpus", document_class=Fact, **kwargs):
        connections.create_connection(hosts=["localhost"])
        document_class.init()

        documents = (
            document_class(meta={"id": id}, fact=doc["fact"]).to_dict(True)
            for id, doc in corpus.items()
        )

        logger.info(f"Building corpus index for {index_name}")

        # RayExecutor().run(documents, self.save_data, {})

        for success, info in tqdm(
            parallel_bulk(
                connections.get_connection(),
                documents,
                thread_count=kwargs.pop("batch_size", multiprocessing.cpu_count()),
                chunk_size=100000,
                max_chunk_bytes=2 * 1024 ** 3,
            )
        ):
            if not success:
                logger.error(f"A document failed: {info} ")

        logger.success("Elastic index successfully built")

        return index_name
