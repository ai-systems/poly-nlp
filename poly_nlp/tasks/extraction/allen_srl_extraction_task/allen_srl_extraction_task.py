import math
import re
from typing import Dict, List

import nltk
import torch
from allennlp.predictors.predictor import Predictor
from loguru import logger
from overrides import overrides
from prefect import Task
from tqdm import tqdm

from covid_fact_bank.utils.data_manipulation.data_manipulation import create_dict_chunks


class AllenSRLExtraction(Task):
    @overrides
    def run(
        self, sentences: Dict[str, str], srl_model_path: str, batch_size: int = 8
    ) -> Dict[str, Dict]:
        """Run SRL extraction
        
        Args:
            sentences (Dict[str,str]): id: {setence}
        
        Returns:
            Dict[str, Dict]: return output id: outpu from allensrl
        """
        logger.info("Running AlleSRL extraction")

        logger.info("Initializing AllenSRL predictor")

        if torch.cuda.is_available():
            logger.info("GPU found")
            logger.info("Initializing Coreference predictor with GPU")
            predictor = Predictor.from_path(srl_model_path, cuda_device=0)
        else:
            logger.info("Initializing Coreference predictor with CPU")
            predictor = Predictor.from_path(srl_model_path)

        logger.info("Running SRL")
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
        logger.success("SRL resolution successful")

        results = {}

        for sentence_id, res_srl in tqdm(resolved_values.items()):
            results[sentence_id] = []
            # if not parsed by the srl
            if len(res_srl["verbs"]) == 0 and not sentences[sentence_id].isspace():
                pos_tags = nltk.pos_tag(nltk.word_tokenize(sentences[sentence_id]))
                found_vb = False
                found_jj = False
                regex = ""
                for word, pos in pos_tags:
                    if "VB" in pos:
                        regex = regex + " " + word
                        found_vb = True
                    elif found_vb:
                        if "JJ" in pos or "RB" in pos or "IN" in pos or "TO" in pos:
                            regex = regex + " " + word
                        else:
                            break
                regex = regex + " "
                pattern = re.findall(regex, " " + sentences[sentence_id] + " ")
                reg_res = re.split(regex, " " + sentences[sentence_id] + " ")
                second_split = reg_res[1]
                if len(reg_res) > 1:
                    if second_split.replace(" ", "").replace(".", "") == "":
                        results[sentence_id].append(
                            "[ARG0: "
                            + reg_res[0][1:]
                            + "] "
                            + "[CV: "
                            + pattern[0][1:-1]
                            + "]"
                        )
                    else:
                        results[sentence_id].append(
                            "[ARG0: "
                            + reg_res[0][1:]
                            + "] "
                            + "[CV: "
                            + pattern[0][1:-1]
                            + "] "
                            + "[ARG1: "
                            + " ".join(reg_res[1:])[:-1]
                            + "]"
                        )
            else:
                # append the detected verbs
                for verb in res_srl["verbs"]:
                    results[sentence_id].append(verb["description"])

        logger.info("SRL completed")

        return results
