import math
import re
from functools import reduce
from typing import Dict, List

import nltk
import numpy as np
import torch

# import allennlp_models.coref.coref_predictor
from allennlp.predictors.predictor import Predictor
from loguru import logger
from overrides import overrides
from prefect import Task
from tqdm.contrib import tqdm

from covid_fact_bank.utils.data_manipulation.data_manipulation import create_dict_chunks


class CoreferenceResolution(Task):
    def coref_sub(self, document, clusters):
        offset = 0
        cluster_max = []
        for cluster in clusters:
            flat = reduce(list.__add__, cluster)
            cluster_max.append(np.max(flat))

        clusters = [clusters[i] for i in np.argsort(cluster_max)]

        for cluster in clusters:
            start_ref, end_ref = cluster[0]
            start_ref += offset
            end_ref += offset
            for coref in cluster[1:]:
                start_coref, end_coref = coref
                start_coref += offset
                end_coref += offset
                offset += (end_ref - start_ref) - (end_coref - start_coref)
                document[start_coref : end_coref + 1] = document[
                    start_ref : end_ref + 1
                ]
        return document

    @overrides
    def run(
        self, paragraphs: Dict[str, str], coref_model_path: str, batch_size=8
    ) -> Dict[str, str]:
        """Run SRL extraction
        
        Args:
            paragraphs (Dict[str,str]): id: {paragraph}
        
        Returns:
            Dict[str, str]: return output id: paragraph with coreference resolution
        """

        if torch.cuda.is_available():
            logger.info("GPU found")
            logger.info("Initializing Coreference predictor with GPU")
            predictor = Predictor.from_path(coref_model_path, cuda_device=0)
        else:
            logger.info("Initializing Coreference predictor with CPU")
            predictor = Predictor.from_path(coref_model_path)

        logger.info(f"Batch_size = {batch_size}")
        batches = create_dict_chunks(paragraphs, batch_size)

        resolved_values = {}
        for batch in tqdm(
            batches,
            desc="Running coreference resolution",
            total=math.ceil(len(paragraphs) / batch_size),
        ):
            resolved_values = {
                **resolved_values,
                **{
                    id: val
                    for id, val in zip(
                        batch.keys(),
                        predictor.predict_batch_json(
                            [{"document": paragraph} for paragraph in batch.values()]
                        ),
                    )
                },
            }
        logger.success("Coreference resolution successful")

        logger.info("Resolving Coreference")
        results = {}

        for key, res in resolved_values.items():
            new_paragraph = self.coref_sub(res["document"], res["clusters"])
            results[key] = " ".join(new_paragraph)

        return results
