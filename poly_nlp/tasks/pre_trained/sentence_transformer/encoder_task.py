from loguru import logger
from overrides import overrides
from prefect import Task
from sentence_transformers import SentenceTransformer


class SentenceTransformerEncoderTask(Task):
    @overrides
    def run(
        self,
        sentences,
        model_name_or_path="roberta-large-nli-stsb-mean-tokens",
        cuda=True,
        **kwargs,
    ):
        logger.info(f"Running Sentence Transformer Task: {model_name_or_path}")
        model = SentenceTransformer(model_name_or_path)
        if cuda:
            logger.info("Running model on GPU")
            model.cuda()
        sentence_embeddings = model.encode(
            list(sentences.values()),
            show_progress_bar=kwargs.get("show_progress_bar", True),
            batch_size=kwargs.get("batch_size", 8),
            convert_to_numpy=kwargs.get("covert_to_numpy", True),
        )
        return {id: sentence_embeddings[index] for index, id in enumerate(sentences)}
