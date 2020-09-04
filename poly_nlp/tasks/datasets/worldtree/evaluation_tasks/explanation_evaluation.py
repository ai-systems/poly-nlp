import math
import string
from collections import defaultdict
from functools import reduce
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import spacy
from dynaconf import settings
from loguru import logger
from nltk.corpus import stopwords
from overrides import overrides
from plotly.offline import plot
from prefect.tasks import Task
from tqdm import tqdm

from poly_nlp.evaluate.metrics import MAP, NDCG, PrecisionTopK, RecallTopK
from poly_nlp.evaluate.visualise.latex import LatexDoc


class ExplanationEvaluationTask(Task):
    def __init__(
        self,
        eb_bank: Dict[str, Dict],
        table_store: Dict[str, Dict],
        skip_overlap_test=False,
    ):
        self.eb_bank = eb_bank
        self.table_store = table_store
        self.skip_overlap_test = skip_overlap_test

        if not skip_overlap_test:
            table = str.maketrans("", "", string.punctuation)
            nlp = spacy.load("en_core_web_sm")
            self.question_tokens = {}
            self.filtered_question_tokens = {}
            for question_instance in tqdm(
                self.eb_bank.values(), desc="Tokenizing question"
            ):
                question_answer = f"{question_instance['question']} {question_instance['answer']}".translate(
                    table
                ).lower()
                docs = nlp(question_answer)
                self.question_tokens[question_instance["id"]] = [
                    doc.text for doc in docs
                ]
                self.filtered_question_tokens[question_instance["id"]] = [
                    doc.text
                    for doc in docs
                    if doc.pos_ in ["NOUN", "VERB", "ADV", "ADJ"]
                ]

            self.table_store_tokens = defaultdict(lambda: [])
            self.filtered_table_store_tokens = defaultdict(lambda: [])
            for table_instance in tqdm(
                self.table_store.values(), desc="Tokenizing facts"
            ):
                docs = nlp(table_instance["fact"].translate(table).lower())
                token_text = [doc.text for doc in docs]
                filtered_token_text = [
                    doc.text
                    for doc in docs
                    if doc.pos_ in ["NOUN", "VERB", "ADV", "ADJ"]
                ]
                self.table_store_tokens[table_instance["id"]] = (
                    self.table_store_tokens[table_instance["id"]] + token_text
                )
                self.filtered_table_store_tokens[table_instance["id"]] = (
                    self.filtered_table_store_tokens[table_instance["id"]]
                    + filtered_token_text
                )

    @overrides
    def run(
        self, predicted: Dict[str, List[str]], table_store_categories: str,
    ):
        table_categories = defaultdict(lambda: [])
        # Get table store categories
        with open(table_store_categories) as f:
            for line in f:
                line = line.strip()
                table_name, category = line.split()
                table_categories[category.split("/")[0]].append(table_name)

        actual: Dict[str, List[str]] = reduce(
            lambda dt, exp: {**dt, exp["id"]: list(exp["explanation"].keys())},
            self.eb_bank.values(),
            {},
        )
        map_overall = MAP(actual, predicted).calc_metric()
        filter_by_role = lambda role: reduce(
            lambda dt, exp: {
                **dt,
                exp["id"]: [
                    key
                    for key, exp_role in exp["explanation"].items()
                    if exp_role == role
                ],
            },
            self.eb_bank.values(),
            {},
        )
        map_by_role = {
            "central": MAP(filter_by_role("CENTRAL"), predicted).calc_metric(),
            "grounding": MAP(filter_by_role("GROUNDING"), predicted).calc_metric(),
            "lex_glues": MAP(filter_by_role("LEXGLUE"), predicted).calc_metric(),
        }
        table_id_name_map = defaultdict(
            lambda: "",
            reduce(
                lambda d, val: {val["id"]: val["table_name"], **d},
                self.table_store.values(),
                {},
            ),
        )

        filter_by_kt = lambda kt_vals: reduce(
            lambda dt, exp: {
                **dt,
                exp["id"]: [
                    key
                    for key, exp_role in exp["explanation"].items()
                    if table_id_name_map[key] in kt_vals
                ],
            },
            self.eb_bank.values(),
            {},
        )

        map_by_ktypes = {}
        for category, kt_vals in table_categories.items():
            map_by_ktypes[category] = MAP(
                filter_by_kt(kt_vals), predicted
            ).calc_metric()

        precision_topk = {}
        for k in [1, 2, 3, 4, 5, 10, 20]:
            precision_topk[k] = PrecisionTopK(actual, predicted, k=k).calc_metric()

        recall_topk_by_category = defaultdict(lambda: {})
        for category, kt_vals in table_categories.items():
            for k in [3, 5, 10]:
                recall_topk_by_category[category][k] = RecallTopK(
                    filter_by_kt(kt_vals), predicted, k=k
                ).calc_metric()

        max_exp_length = reduce(
            lambda max_len, exp: max(max_len, len(exp["explanation"])),
            self.eb_bank.values(),
            0,
        )

        filter_by_exp_len = lambda exp_len: reduce(
            lambda dt, exp: {**dt, exp[0]: exp[1]},
            [
                (exp["id"], list(exp["explanation"].keys()))
                for exp in self.eb_bank.values()
                if (exp_len <= 10 and len(exp["explanation"]) == exp_len)
                or len(exp["explanation"]) > 10
            ],
            {},
        )

        filter_by_exp_len_kt = lambda exp_len: reduce(
            lambda dt, exp: {**dt, exp[0]: exp[1]},
            [
                (id, exps)
                for id, exps in filter_by_kt(table_categories["COMPLEX"]).items()
                if (exp_len <= 10 and len(exps) == exp_len) or len(exps) > 10
            ],
            {},
        )

        map_by_exp_len = {}
        map_by_exp_len_ticks = {}
        for i in range(1, 12):
            map_score = MAP(filter_by_exp_len(i), predicted).calc_metric()
            if not math.isnan(map_score):
                map_by_exp_len[i] = map_score
                map_by_exp_len_ticks[i] = f"{i}" if i <= 10 else "10+"

        filter_by_hops = lambda overlap, f_table_tokens, f_ques_tokens: reduce(
            lambda dt, exp: {
                **dt,
                exp["id"]: [
                    key
                    for key, exp_role in exp["explanation"].items()
                    if overlap(
                        len((set(f_table_tokens[key]) & set(f_ques_tokens[exp["id"]])))
                    )
                ],
            },
            self.eb_bank.values(),
            {},
        )
        map_by_hops_filtered = {}
        map_by_hops_not_filtered = {}
        if not self.skip_overlap_test:
            map_by_hops_cond = {
                "1_hop (2)": lambda val: val >= 2,
                "1_hop (1)": lambda val: val == 2,
                "2hops": lambda val: val == 0,
            }
            for key, cond in map_by_hops_cond.items():
                map_by_hops_filtered[key] = MAP(
                    filter_by_hops(cond, self.table_store_tokens, self.question_tokens),
                    predicted,
                ).calc_metric()

            for key, cond in map_by_hops_cond.items():
                map_by_hops_not_filtered[key] = MAP(
                    filter_by_hops(
                        cond,
                        self.filtered_table_store_tokens,
                        self.filtered_question_tokens,
                    ),
                    predicted,
                ).calc_metric()

        return {
            "map_overall": map_overall,
            "map_by_explanatory_role": map_by_role,
            "map_by_knowledge_types": map_by_ktypes,
            "precision_topk": precision_topk,
            "recall_topk_by_category": dict(recall_topk_by_category),
            "map_by_exp_len": map_by_exp_len,
            "map_by_exp_len_ticks": map_by_exp_len_ticks,
            "map_by_hops_filtered": map_by_hops_filtered,
            "map_by_hops_not_filtered": map_by_hops_not_filtered,
        }
