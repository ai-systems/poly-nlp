import math
from enum import Enum

from loguru import logger
from overrides import overrides
from prefect import Task
from sentence_transformers import (
    InputExample,
    SentencesDataset,
    SentenceTransformer,
    losses,
    models,
)
from torch.utils.data import DataLoader


class SentenceTransformerLoss(Enum):
    cosine_similarity_loss = losses.CosineSimilarityLoss
    mse_loss = losses.MSELoss


class SentenceTransformerTrainerTask(Task):
    @overrides
    def run(
        self,
        training_data,
        evaluator,
        output_path,
        from_scratch=False,
        loss=SentenceTransformerLoss.cosine_similarity_loss,
        model_name_or_path="roberta-large-nli-stsb-mean-tokens",
        cuda=True,
        **kwargs,
    ):
        logger.info(
            f"Running Sentence Transformer Task: {model_name_or_path}, Output path: {output_path}"
        )
        if from_scratch:
            logger.info("Training from scratch")
            models.Transformer(
                model_name_or_path, max_seq_length=kwargs.get("max_seq_length", 128)
            )
        else:
            model = SentenceTransformer(model_name_or_path)
        if cuda:
            logger.info("Running model on GPU")
            model.cuda()

        train_examples = [
            InputExample(
                texts=[data["sentence1"], data["sentence2"]], label=data["label"]
            )
            for data in training_data.values()
        ]
        train_dataset = SentencesDataset(train_examples, model)
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=kwargs.get("shuffle", True),
            batch_size=kwargs.get("batch_size", 4),
        )
        warmup_steps = math.ceil(
            len(train_examples)
            * kwargs.get("num_epochs", 3)
            / kwargs.get("train_batch_size", 4)
            * 0.1
        )  # 10% of train data for warm-up
        train_loss = loss.value(model)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=kwargs.get("num_epochs", 3),
            evaluation_steps=kwargs.get("evaluation_steps", 500),
            warmup_steps=warmup_steps,
            output_path=output_path,
            evaluator=evaluator,
        )
