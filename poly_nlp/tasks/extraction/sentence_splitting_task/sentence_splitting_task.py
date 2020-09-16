from typing import Dict
from uuid import uuid4

import spacy
from prefect import Task
from tqdm import tqdm

from poly_nlp.parallel.ray_executor import RayExecutor


class SentenceSplitTask(Task):
    @staticmethod
    def process_sentences(pos, input):
        split_sentences = {}
        nlp = spacy.load("en_core_web_sm")
        for id, sentences in input.items():
            text_sentences = nlp(sentences)
            split_sentences[id] = {}
            split_sentences[id] = [sent.text for sent in text_sentences.sents]
        return split_sentences

    def run(self, paragraph: Dict[str, str]):
        return RayExecutor().run(paragraph, self.process_sentences, {})
