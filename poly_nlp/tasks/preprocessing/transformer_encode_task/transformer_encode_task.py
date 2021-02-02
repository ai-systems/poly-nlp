import csv
import os
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
from transformers import AutoTokenizer

from .transformer_utils import encode_sentence_pairs


class TransformerEncodeTask(Task):
    @staticmethod
    def tokenize(input, output_path, maxlen, transformer_model, pos=0):
        query_mapping = {}
        input_ids = np.memmap(
            f"{output_path}/input_ids_{pos}.mmap",
            dtype="long",
            mode="w+",
            shape=(len(input), maxlen),
        )
        attention_masks = np.memmap(
            f"{output_path}/attention_maps_{pos}.mmap",
            dtype="long",
            mode="w+",
            shape=(len(input), maxlen),
        )
        tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        for index, (id, query) in enumerate(
            tqdm(input.items(), "Tokenizing and encoding to transformer format")
        ):
            query_mapping[id] = (pos, index)

            # TODO: Need to map tokentype_ids
            if isinstance(query, str):
                i_ids, a_ids, t_ids = encode_sentence_pairs(
                    sentence1=query, tokenizer=tokenizer, max_length=maxlen
                )
            else:
                i_ids, a_ids, t_ids = encode_sentence_pairs(
                    sentence1=query["sentence1"],
                    sentence2=query["sentence2"],
                    tokenizer=tokenizer,
                    max_length=maxlen,
                )

            input_ids[index] = i_ids
            attention_masks[index] = a_ids
        return {
            pos: {
                "input_ids": np.memmap(
                    f"{output_path}/input_ids_{pos}.mmap",
                    dtype="long",
                    mode="r+",
                    shape=(len(input), maxlen),
                ),
                "attention_masks": np.memmap(
                    f"{output_path}/attention_maps_{pos}.mmap",
                    dtype="long",
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
        t_name,
        transformer_model,
        maxlen=128,
    ):
        logger.info("Tokenizing text")
        output_path = os.path.join(output_path, t_name)
        if not os.path.exists(output_path):
            logger.info(f"{output_path} not exists. Creating a new one")
            os.makedirs(output_path)
        vocab_mapped_text = self.tokenize(
            input=text_input,
            output_path=output_path,
            maxlen=maxlen,
            transformer_model=transformer_model,
        )

        combined_query_mapping = vocab_mapped_text[0]["query_mapping"]

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
            )
        }
