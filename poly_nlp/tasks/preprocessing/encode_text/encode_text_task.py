import csv
import os
from collections.abc import MutableMapping
from functools import reduce
from typing import Dict

import numpy as np
import pandas as pd
import spacy
from loguru import logger
from overrides import overrides
from poly_nlp.parallel.ray_executor import RayExecutor
from poly_nlp.utils.data_structures.transformed_dict import TransformedDict
from prefect import Task
from tqdm import tqdm


class EncodeTextTask(Task):
    @staticmethod
    def tokenize(pos, input):
        nlp = spacy.load("en_core_web_sm")
        query_output = {}
        for id, query in input.items():
            query_output[id] = [word.text for word in nlp(query) if word.text != " "]
        return query_output

    @staticmethod
    def map_vocab(pos, input, vec, maxlen, dtype, padding, truncating, output_path):
        query_mapping = {}
        input_ids = np.memmap(
            f"{output_path}/input_ids_{pos}.mmap",
            dtype=dtype,
            mode="w+",
            shape=(len(input), maxlen),
        )
        attention_masks = np.memmap(
            f"{output_path}/attention_maps_{pos}.mmap",
            dtype=dtype,
            mode="w+",
            shape=(len(input), maxlen),
        )

        for index, (id, query) in enumerate(
            tqdm(input.items(), f"Mapping vocab", position=pos, disable=True)
        ):
            if len(query) > maxlen:
                print(f"WARNING: {id} is greater than maximum length. Truncating")
            query_mapping[id] = (pos, index)
            input_ids[index] = [
                vec(query[index].lower()) if index < len(query) else 0
                for index in range(0, maxlen)
            ]
            attention_masks[index] = [
                1 if index < len(query) else 0 for index in range(0, maxlen)
            ]

        return {
            pos: {
                "input_ids": np.memmap(
                    f"{output_path}/input_ids_{pos}.mmap",
                    dtype=dtype,
                    mode="r+",
                    shape=(len(input), maxlen),
                ),
                "attention_masks": np.memmap(
                    f"{output_path}/attention_maps_{pos}.mmap",
                    dtype=dtype,
                    mode="r+",
                    shape=(len(input), maxlen),
                ),
                "query_mapping": query_mapping,
            }
        }

    @overrides
    def run(
        self,
        text_input,
        output_path,
        task_name,
        maxlen=128,
        dtype="int32",
        padding="post",
        truncating="post",
        vocab={},
        pretrained_file=None,
        extend_vocab=True,
    ):
        logger.info("Tokenizing text")
        output_path = os.path.join(output_path, task_name)
        if not os.path.exists(output_path):
            logger.info(f"{output_path} not exists. Creating a new one")
            os.makedirs(output_path)
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

        vocab_mapped_text = ray_executor.run(
            tokenized_output,
            self.map_vocab,
            dict(
                vec=vec,
                maxlen=maxlen,
                output_path=output_path,
                dtype=dtype,
                padding=padding,
                truncating=truncating,
            ),
        )
        combined_query_mapping = reduce(
            lambda x, y: {**x, **y["query_mapping"]}, vocab_mapped_text.values(), {}
        )
        if pretrained_file is not None:
            glove_vectors = words.to_numpy()
            return {
                "inputs": TransformedDict(
                    combined_query_mapping,
                    [
                        (
                            key,
                            {k: v for k, v in val.items() if not k == "query_mapping"},
                        )
                        for key, val in vocab_mapped_text.items()
                    ],
                ),
                "embedding": np.vstack(
                    (np.zeros_like(glove_vectors[0]), glove_vectors)
                ),
            }
        else:
            return {"inputs": vocab_mapped_text}
