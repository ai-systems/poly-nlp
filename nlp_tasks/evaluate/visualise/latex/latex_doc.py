from collections import defaultdict
from typing import Dict, List

from loguru import logger
from pylatex import Document, Section
from .metric_table import MetricTable

from ...metrics import Metric


class LatexDoc:
    @staticmethod
    def build_latex(
        evaluations: Dict[str, List[Metric]],
        file_name: str,
        invert_map: Dict[str, bool] = defaultdict(lambda: False),
        **kwargs,
    ):
        logger.info("Building LaTex Document")
        doc = Document("")
        section = Section("Evaluations")
        tables = [
            MetricTable.build_latex(evals, invert=invert_map[caption], caption=caption)
            for caption, evals in evaluations.items()
        ]
        doc.append(section)
        for table in tables:
            section.append(table)
        doc.generate_tex(filepath=file_name)
        logger.success("LaTex Document build success")
        return doc

    @staticmethod
    def build_pdf(
        evaluations: Dict[str, List[Metric]],
        file_name: str,
        invert_map: Dict[str, bool] = defaultdict(lambda: False),
        **kwargs,
    ):
        logger.info("Building PDF File")
        doc: Document = LatexDoc.build_latex(evaluations, file_name, invert_map)
        doc.generate_pdf(filepath=file_name)
        logger.success("PDF build succesful")
