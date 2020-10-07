import csv
from typing import Dict

import numpy as np
import pandas as pd
import spacy
from loguru import logger
from overrides import overrides
from poly_nlp.parallel.ray_executor import RayExecutor
from prefect import Task
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
    def map_vocab(input, vec, maxlen, dtype, padding, truncating, output_path):
        query_mapping = {}
        input_ids = np.memmap(
            f"{output_path}/input_ids.mmap",
            dtype=dtype,
            mode="w+",
            shape=(len(input), maxlen),
        )
        attention_masks = np.memmap(
            f"{output_path}/attention_maps.mmap",
            dtype=dtype,
            mode="w+",
            shape=(len(input), maxlen),
        )

        for index, (id, query) in enumerate(tqdm(input.items(), "Mapping vocab")):
            if len(query) > maxlen:
                logger.warning(f"{id} is greater than maximum length. Truncating")
            query_mapping[id] = index
            input_ids[index] = [
                vec(query[index].lower()) if index < len(query) else 0
                for index in range(0, maxlen)
            ]
            attention_masks[index] = [
                1 if index < len(query) else 0 for index in range(0, maxlen)
            ]

        return {
            "input_ids": np.memmap(
                f"{output_path}/input_ids.mmap",
                dtype=dtype,
                mode="r+",
                shape=(len(input), maxlen),
            ),
            "attention_masks": np.memmap(
                f"{output_path}/attention_maps.mmap",
                dtype=dtype,
                mode="r+",
                shape=(len(input), maxlen),
            ),
            "query_mapping": query_mapping,
        }

    @overrides
    def run(
        self,
        text_input,
        output_path,
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
                lambda w: words.index.get_loc(w) + 1
                if w in words.index
                else words.index.get_loc("unk") + 1
            )
        else:
            raise NotImplementedError("Pretrained encoding only implemented")

        vocab_mapped_text = self.map_vocab(
            input=tokenized_output,
            vec=vec,
            maxlen=maxlen,
            output_path=output_path,
            dtype=dtype,
            padding=padding,
            truncating=truncating,
        )
        if pretrained_file is not None:
            glove_vectors = words.to_numpy()
            return {
                "inputs": vocab_mapped_text,
                "embedding": np.vstack(
                    (np.zeros_like(glove_vectors[0]), glove_vectors)
                ),
            }
        else:
            return {"inputs": vocab_mapped_text}

