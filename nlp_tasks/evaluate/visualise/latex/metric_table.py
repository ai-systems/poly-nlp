from typing import List
from collections import defaultdict
from ...metrics import Metric, MODEL_NAME, SPLIT_NAME, METRIC_LABEL, SCORE
from pylatex import Tabular, MultiColumn, MultiRow, Table
from itertools import cycle, zip_longest
import math


class MetricTable:
    @staticmethod
    def build_latex(metrics: List[Metric], invert=False, caption=""):
        models, metric_labels, splits = {}, {}, {}
        score_map = defaultdict(lambda: "-")
        for metric in metrics:
            metric_vals = metric()
            score, metric_label, model_name, split_name = (
                metric_vals[SCORE],
                metric_vals[METRIC_LABEL],
                metric_vals[MODEL_NAME],
                metric_vals[SPLIT_NAME],
            )
            metric_labels[metric_label] = 1
            models[model_name] = 1
            splits[split_name] = 1
            score_map[(metric_label, model_name, split_name)] = (
                "-" if math.isnan(score) else score
            )

        models = list(models.keys())
        splits = list(splits.keys())
        metric_labels = list(metric_labels.keys())

        if not invert:
            model_split_cols = "".join(["c|" for _ in range(len(models) * len(splits))])
            table = Tabular("|c|" + model_split_cols)
            table.add_hline()
            row = (MultiRow(2, data="Metric"),) if len(splits) > 1 else ("Metric",)
            for model_name in models:
                row += (MultiColumn(len(splits), align="|c|", data=model_name),)
            table.add_row(row)
            table.add_hline(1 if len(splits) == 1 else 2)
            if (len(splits)) > 1:
                row = ("",)
                for _ in range(len(models)):
                    for split_name in splits:
                        row += (split_name,)
                table.add_row(row)
                table.add_hline()
            for metric_label in metric_labels:
                row = (metric_label,)
                for model_name in models:
                    for split_name in splits:
                        row += (score_map[(metric_label, model_name, split_name)],)
                table.add_row(row)
                table.add_hline()
        else:
            metric_split_labels = "".join(["c|" for _ in range(len(metric_labels))])
            table = Tabular(
                "|c|" if len(splits) == 1 else "|c|c|" + metric_split_labels
            )
            table.add_hline()
            row = (MultiColumn(2, align="|c|", data=""),) if len(splits) > 1 else ("",)
            for metric_label in metric_labels:
                row += (metric_label,)
            table.add_row(row)
            table.add_hline()
            for model_name in models:
                row = (MultiRow(len(splits), data=model_name),)
                for split_name in splits:
                    row += (split_name,)
                    for metric_label in metric_labels:
                        row += (score_map[(metric_label, model_name, split_name)],)
                    table.add_row(row)
                    if len(splits) > 1:
                        table.add_hline(start=2)
                    row = ("",)
                table.add_hline()

        latex_table = Table(position="h")
        latex_table.append(table)
        latex_table.add_caption(caption)
        return latex_table

