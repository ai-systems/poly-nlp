import math
import re
from typing import Dict, List

import nltk
import torch
from allennlp.predictors.predictor import Predictor
from loguru import logger
from overrides import overrides
from prefect import Task
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class SentenceTransformerEmbeddingTask(Task):
    @overrides
    def run(
        self,
        sentences: Dict[str, str],
        model_name: str = "roberta-base-nli-stsb-mean-tokens",
        batch_size: int = 8,
    ) -> Dict[str, Dict]:
        logger.info(f"Running sentence transformers, Model name: {model_name}")
        model = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            logger.info("GPU found")
            logger.info("Initializing Coreference predictor with GPU")
            model.cuda()
        else:
            logger.info("Initializing Coreference predictor with CPU")
        sentence_embeddings = model.encode(
            list(sentences.values()), batch_size=batch_size, show_progress_bar=True
        )
        return {
            sentence_id: embedding
            for sentence_id, embedding in zip(sentences.keys(), sentence_embeddings)
        }
