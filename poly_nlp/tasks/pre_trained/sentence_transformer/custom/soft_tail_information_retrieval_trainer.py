from collections import defaultdict
from enum import Enum
from functools import reduce

from loguru import logger
from overrides import overrides
from poly_nlp.tasks.pre_trained.sentence_transformer import (
    SentenceTransformerEncoderTask,
    SentenceTransformerTrainerTask,
)
from poly_nlp.tasks.retrieval.faiss import FaissIndexBuildTask, FaissSearchTask
from prefect import Flow, Task
from scipy.spatial import distance
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from tqdm import tqdm


class SoftTailIRTrainer(Task):
    @overrides
    def run(
        self,
        train_data,
        dev_data,
        output_path,
        model_name_or_path="roberta-large-nli-stsb-mean-tokens",
        soft_tail=5,
        test_data=None,
        faiss_opts={"mips": True},
        **kwargs,
    ):
        queries_train, corpus_train, relevant_docs_train = train_data
        queries_dev, corpus_dev, relevant_docs_dev = dev_data
        if test_data is not None:
            logger.warning("Test data evaluation is not implemented yet")
            queries_test, corpus_test, relevant_docs_test = test_data

        logger.debug("Setting up relevant train query")
        relevant_train_query = reduce(
            lambda query_set, items: query_set.union(set(items)),
            relevant_docs_train.values(),
            set(),
        )
        relevant_train_query_map = {
            f_id: corpus_train[f_id]
            for f_id in relevant_train_query
            if f_id in corpus_train
        }
        if not len(relevant_train_query) == len(relevant_train_query_map):
            logger.warning(
                f"Number of query: {len(relevant_train_query)} and Number of query map: {len(relevant_train_query_map)}. Length mismatch"
            )

        encoder_task = SentenceTransformerEncoderTask()
        indexing_task = FaissIndexBuildTask()
        search_task = FaissSearchTask()

        with Flow("Faiss Task") as faiss_flow:
            corpus_embeddings = encoder_task(corpus_train)
            c_query_embeddings = encoder_task(relevant_train_query_map)
            faiss_index = indexing_task(corpus_embeddings, faiss_opts)
            nearest_indexes = search_task(faiss_index, c_query_embeddings, k=soft_tail)

        state = faiss_flow.run()
        nearest_neighbors_train = state.result[nearest_indexes]._result.value
        corpus_embeddings_train = state.result[corpus_embeddings]._result.value

        corpus_train_index = {
            index: val for index, val in enumerate(corpus_train.values())
        }
        corpus_train_id = {index: val for index, val in enumerate(corpus_train.keys())}

        # Creating softail data
        soft_tail_data = defaultdict(lambda: [])
        for id, result in nearest_neighbors_train.items():
            for index in result["index"]:
                if index in corpus_train_index and corpus_train_id[index] != id:
                    dist = distance.cosine(
                        corpus_embeddings_train[corpus_train_id[index]],
                        corpus_embeddings_train[id],
                    )
                    soft_tail_data[id].append((corpus_train_id[index], dist))

        s_transformer_train_data = {}
        positive_count = 0
        for d_id, relevant_doc_ids in tqdm(
            relevant_docs_train.items(), "Prepraring training data"
        ):
            if d_id not in queries_train:
                logger.warning(f"{d_id} not found in queries train")
                continue
            for rel_id in relevant_doc_ids:
                if rel_id not in corpus_train:
                    logger.warning(f"{rel_id} not found in corpus train")
                    continue
                s_transformer_train_data[f"{d_id}|{rel_id}"] = {
                    "sentence1": queries_train[d_id],
                    "sentence2": corpus_train[rel_id],
                    "label": 1.0,
                }
                positive_count += 1
                for (tail_rel_id, dist) in soft_tail_data[rel_id]:
                    if tail_rel_id not in corpus_train:
                        logger.warning(
                            f"{tail_rel_id} (tail) not found in corpus train"
                        )
                        continue
                    if tail_rel_id in relevant_doc_ids:
                        continue
                    s_transformer_train_data[f"{d_id}|{tail_rel_id}"] = {
                        "sentence1": queries_train[d_id],
                        "sentence2": corpus_train[tail_rel_id],
                        "label": 1 - float(dist),
                        # "label": float(dist),
                    }

        logger.info(
            f"Sentence Transformer train data count: {len(s_transformer_train_data)}"
        )
        logger.info(f"Positive data count: {positive_count}")

        sentence_transformer_trainer = SentenceTransformerTrainerTask()
        dev_evaluator = InformationRetrievalEvaluator(
            queries_dev,
            corpus_dev,
            relevant_docs_dev,
            show_progress_bar=True,
            batch_size=kwargs.get("batch_size", 8),
        )

        with Flow("Training Sentence Transformer") as train_flow:
            trainer = sentence_transformer_trainer(
                s_transformer_train_data,
                evaluator=dev_evaluator,
                output_path=output_path,
                model_name_or_path=kwargs.get(
                    "model_name_or_path", "bert-base-nli-stsb-mean-tokens"
                ),
                num_epochs=kwargs.get("num_epochs", 5),
                train_batch_size=kwargs.get("train_batch_size", 16),
            )

        state = train_flow.run()
