import sys
from collections import defaultdict
from uuid import uuid4

from loguru import logger
from overrides import overrides
from prefect import Task
from tqdm import tqdm


class ConceptNetExtend(Task):
    @staticmethod
    def process_args(arg_map, arg, prev_arg):
        if arg not in arg_map:
            return []
        elif len(arg_map[arg]) == 0:
            return []
        else:
            for args in arg_map[arg]:
                e_arg, e_connector = args
                if e_arg in prev_arg:
                    return []
                prev_arg = prev_arg + e_arg
                return [(e_arg, e_connector)] + ConceptNetExtend.process_args(
                    arg_map, e_arg, prev_arg
                )

    @overrides
    def run(self, data, skip_non_taxonomical=True):
        logger.info("Running conceptnet extender")
        logger.info(f"Data length: {len(data)}")

        arg_map = defaultdict(lambda: [])

        new_conceptnet = {}
        for id, fact_info in tqdm(data.items(), "Mapping arguments"):
            fact = fact_info["fact"].lower()
            if " is " in fact:
                (arg1, arg2), connector = fact.split(" is "), "is"
            elif " are " in fact:
                (arg1, arg2), connector = fact.split(" are "), "are"
            arg2 = arg2.replace(".", "")
            if len(arg2.split()) > 2:
                if not skip_non_taxonomical:
                    new_conceptnet[id] = {}
                    new_conceptnet[id]["fact"] = fact
                continue
            if arg1 == arg2:
                continue
            arg_map[arg1].append((arg2, connector))

        extended_arg_map = {}
        arg_map = dict(arg_map)
        for arg1 in tqdm(arg_map.keys(), "Processing data"):
            extended_arg_map[arg1] = self.process_args(arg_map, arg1, arg1)

        for arg1, extended_data in tqdm(
            extended_arg_map.items(), "Generating extended conceptnet"
        ):
            for arg2, connector in extended_data:
                fact = f"{arg1} {connector} {arg2} ."
                id = str(uuid4())
                new_conceptnet[id] = {}
                new_conceptnet[id]["fact"] = fact
        return new_conceptnet
