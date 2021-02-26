import math
import re
from typing import Dict, List

import allennlp_models.tagging
import nltk
import torch
from allennlp.predictors.predictor import Predictor
from loguru import logger
from overrides import overrides
from poly_nlp.utils.data_manipulation import create_dict_chunks
from prefect import Task
from tqdm import tqdm


class AllenOIEExtractionTask(Task):
    @overrides
    def run(
        self, sentences: Dict[str, str], oie_model_path: str, batch_size: int = 8
    ) -> Dict[str, Dict]:
        """Run SRL extraction

        Args:
            sentences (Dict[str,str]): id: {setence}

        Returns:
            Dict[str, Dict]: return output id: outpu from allensrl
        """
        logger.info("Running AlleOIE extraction")

        if torch.cuda.is_available():
            logger.info("GPU found")
            logger.info("Initializing Coreference predictor with GPU")
            predictor = Predictor.from_path(oie_model_path, cuda_device=0)
        else:
            logger.info("Initializing Coreference predictor with CPU")
            predictor = Predictor.from_path(oie_model_path)

        logger.info("Running OIE")
        logger.info(f"Batch_size = {batch_size}")
        batches = create_dict_chunks(sentences, batch_size)

        resolved_values = {}
        for batch in tqdm(
            batches,
            desc="Running SRL Extraction",
            total=math.ceil(len(sentences) / batch_size),
        ):
            resolved_values = {
                **resolved_values,
                **{
                    id: val
                    for id, val in zip(
                        batch.keys(),
                        predictor.predict_batch_json(
                            [{"sentence": sentence} for sentence in batch.values()]
                        ),
                    )
                },
            }

        results = {}

        for sentence_id, res_srl in tqdm(resolved_values.items()):
            results[sentence_id] = {}
            for index, verb in enumerate(res_srl["verbs"]):
                desc = verb["description"]
                extracted_args = {}
                for group in re.findall(r"[^[]*\[([^]]*)\]", desc):
                    arg_vals = group.split(":")
                    arg_name = arg_vals[0]
                    if len(arg_vals) > 2:
                        argument = " ".join(arg_vals[1:])
                    else:
                        argument = arg_vals[1]
                    arg_name = arg_name.lstrip().strip()
                    argument = argument.lstrip().strip()
                    extracted_args[arg_name] = argument

                results[sentence_id][index] = extracted_args

        logger.info("OIE completed")
        return results
