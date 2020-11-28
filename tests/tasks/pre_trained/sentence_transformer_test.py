import json
import unittest

from poly_nlp.tasks.pre_trained.sentence_transformer import (
    SentenceTransformerEncoderTask,
    SentenceTransformerTrainerTask,
)
from poly_nlp.tasks.pre_trained.sentence_transformer.custom import SoftTailIRTrainer
from sentence_transformers.evaluation import InformationRetrievalEvaluator


class SentenceTransformerTest(unittest.TestCase):
    @unittest.skip
    def test_encoding(self):
        sentences = {
            "id1": "This framework generates embeddings for each input sentence",
            "id2": "Sentences are passed as a list of string.",
            "id3": "The quick brown fox jumps over the lazy dog.",
        }
        output = SentenceTransformerEncoderTask().run(sentences)
        print(output)
        print(output["id1"].shape)

    @unittest.skip
    def test_training(self):
        train_data = {
            "id1": {
                "sentence1": "This is a fact",
                "sentence2": "I agree",
                "label": 1.0,
            },
            "id2": {
                "sentence1": "This is not a fact",
                "sentence2": "I do not agree",
                "label": 0.0,
            },
        }

        corpus = {"id1": "I am most inclinated", "id2": "I am not inclinated"}
        queries = {"qid": "Is this a fact"}
        relevant_docs = {"qid": ["id1"]}
        evaluator = InformationRetrievalEvaluator(
            queries, corpus, relevant_docs, show_progress_bar=True, batch_size=8
        )
        SentenceTransformerTrainerTask().run(
            train_data, evaluator=evaluator, output_path="./data/temp"
        )

    # @unittest.skip
    def test_soft_tail(self):
        with open("data/tests/corpus.json") as f:
            corpus = json.load(f)

        with open("data/tests/explanations_dev.json") as f:
            explanations = json.load(f)

        with open("data/tests/queries_dev.json") as f:
            queries = json.load(f)

        # corpus = {
        #     id: val for index, (id, val) in enumerate(corpus.items()) if index < 100
        # }

        # explanations = {
        #     id: val
        #     for index, (id, val) in enumerate(explanations.items())
        #     if index < 100
        # }

        # queries = {
        #     id: val for index, (id, val) in enumerate(queries.items()) if index < 100
        # }

        dev_data = queries, corpus, explanations

        SoftTailIRTrainer().run(
            train_data=(queries, corpus, explanations),
            dev_data=dev_data,
            output_path="./data/temp",
        )
