import csv
from typing import Dict

import numpy as np
import pandas as pd
import spacy
from loguru import logger
from overrides import overrides
from poly_nlp.parallel.ray_executor import RayExecutor
from prefect import Task
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm


class EncodeTextTask(Task):
    @staticmethod
    def tokenize(pos, input):
        nlp = spacy.load("en_core_web_sm")
        query_output = {}
        for id, query in input.items():
            query_output[id] = [word.text for word in nlp(query)]
        return query_output

    @staticmethod
    def map_vocab(input, vec, maxlen, dtype, padding, truncating):
        query_mapping = {}
        input_ids = np.empty(len(input), dtype=object)
        attention_masks = np.empty(len(input), dtype=object)

        for index, (id, query) in enumerate(tqdm(input.items(), "Mapping vocab")):
            query_mapping[index] = id
            input_ids[index] = [vec(word.lower()) for word in query]
            attention_masks[index] = [1 for word in query]

        input_ids = pad_sequences(
            input_ids,
            maxlen=maxlen,
            dtype=dtype,
            padding=padding,
            truncating=truncating,
        )
        attention_masks = pad_sequences(
            attention_masks,
            maxlen=maxlen,
            dtype=dtype,
            padding=padding,
            truncating=truncating,
        )
        return {
            "input_ids": input_ids,
            "attention_masks": attention_masks,
            "query_mapping": query_mapping,
        }

    @overrides
    def run(
        self,
        text_input,
        maxlen=128,
        dtype="int32",
        padding="post",
        truncating="post",
        vocab={},
        pretrained_file=None,
        extend_vocab=True,
    ):
        logger.info("Tokenizing text")
        ray_executor = RayExecutor()
        tokenized_output = ray_executor.run(text_input, self.tokenize, {})

        if pretrained_file is not None:
            logger.info(f"Loading pretrained embedding file from {pretrained_file}")
            words = pd.read_table(
                pretrained_file,
                sep=" ",
                index_col=0,
                header=None,
                quoting=csv.QUOTE_NONE,
            )
            logger.info("Pretrained file loaded")
            extend_vocab = False
            vec = (
                lambda w: words.index.get_loc(w)
                if w in words.index
                else words.index.get_loc("unk")
            )
        else:
            raise NotImplementedError("Pretrained encoding only implemented")

        vocab_mapped_text = self.map_vocab(
            input=tokenized_output,
            vec=vec,
            maxlen=maxlen,
            dtype=dtype,
            padding=padding,
            truncating=truncating,
        )
        if pretrained_file is not None:
            return {"inputs": vocab_mapped_text, "embedding": words.to_numpy()}
        else:
            return {"inputs": vocab_mapped_text}

