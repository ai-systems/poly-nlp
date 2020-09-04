from typing import Dict
from uuid import uuid4

import spacy
from prefect import Task
from tqdm import tqdm


class SentenceSplitTask(Task):
    def run(self, paragraph: Dict[str, str]):
        nlp = spacy.load("en_core_web_sm")

        split_sentences = {}
        for id, sentences in tqdm(paragraph.items(), desc="Splitting sentences"):
            text_sentences = nlp(sentences)
            split_sentences[id] = {}
            for sent in text_sentences.sents:
                split_sentences[id][str(uuid4())] = sent.text

        return split_sentences
