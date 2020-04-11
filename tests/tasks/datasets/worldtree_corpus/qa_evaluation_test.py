from unittest import TestCase

import numpy as np
import torch
from dynaconf import settings
from prefect import Flow

from regra.tasks.datasets.explanation_bank.data_process_tasks import (
    EBBertProcessingTask,
)
from regra.tasks.datasets.explanation_bank.evaluation_tasks import QAEvaluationTask
from regra.tasks.datasets.explanation_bank.extraction_tasks import (
    ExplanationBankExtraction,
    TableStoreExtractionTask,
)


class QAEvaluationTest(TestCase):
    def test_qa_evaluation(self):
        table_store_path = settings.DATASET.EXPLANATION_BANK.TABLE_STORE_PATH
        expl_bank_data_files = settings.DATASET.EXPLANATION_BANK.DATA["train"]
        qa_evaluation = QAEvaluationTask()

        eb_extraction = ExplanationBankExtraction()
        table_store_extraction = TableStoreExtractionTask()
        bert_extraction = EBBertProcessingTask()
        eb_bank = eb_extraction.run(expl_bank_data_files)
        table_store = table_store_extraction.run(table_store_path)
        datasets = bert_extraction.run(eb_bank, eb_bank, table_store)
        acc = qa_evaluation.evaluate_acc(
            datasets[0],
            eb_bank,
            np.array([i for i in range(len(eb_bank))]),
            np.random.rand(len(eb_bank), 2),
        )
