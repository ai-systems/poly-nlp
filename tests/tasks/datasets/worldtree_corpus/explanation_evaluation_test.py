from collections import defaultdict
from unittest import TestCase

import pandas as pd
import plotly.express as px
from dynaconf import settings
from plotly.offline import plot

from regra.tasks.data_extraction.explanation_bank import (
    ExplanationBankExtraction,
    TableStoreExtractionTask,
)
from regra.tasks.evaluation_task.explanation_bank_evaluation import (
    ExplanationBankEvalTask,
)


class ExperimentEvalTaskTest(TestCase):
    def experiment_eval_test(self):
        predicted = defaultdict(lambda: [])
        with open("./data/results/bm25.txt") as f:
            for line in f:
                line = line.strip()
                id, t_id = line.split()
                predicted[id].append(t_id)
        table_store_path = settings.DATASET.EXPLANATION_BANK.TABLE_STORE_PATH
        table_store_categories_file = settings.DATASET.EXPLANATION_BANK[
            "table_store_categories"
        ]
        expl_bank_data_files = settings.DATASET.EXPLANATION_BANK.DATA["dev"]
        table_store = TableStoreExtractionTask().run(table_store_path)
        explanation_bank = ExplanationBankExtraction().run(expl_bank_data_files)
        eval_task = ExplanationBankEvalTask()
        output = eval_task.run(
            explanation_bank, table_store, predicted, table_store_categories_file
        )
        df1 = pd.DataFrame(
            output["map_by_exp_len"].items(), columns=["Explanation Length", "MAP"]
        )
        df1["models"] = "model1"

        df2 = pd.DataFrame(
            output["map_by_exp_len"].items(), columns=["Explanation Length", "MAP"]
        )
        df2["models"] = "model2"

        df = pd.concat([df1, df2])
        fig = px.line(df, x="Explanation Length", y="MAP", color="models")
        plot(
            fig,
            auto_open=False,
            filename=settings.DATASET.EXPLANATION_BANK.EVALUATION["plot_output"],
        )
