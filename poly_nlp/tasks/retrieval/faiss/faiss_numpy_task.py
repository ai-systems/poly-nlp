from functools import reduce
from typing import Dict

import numpy as np
from loguru import logger
from overrides import overrides
from prefect import Task

import faiss


class FaissIndexBuildTask(Task):
    @overrides
    def run(self, data: Dict[str, np.ndarray], opts={}):
        logger.info("Constructing Faiss")
        data_db = reduce(
            lambda data, val: np.vstack((data, val)) if data is not None else val,
            data.values(),
            None,
        )
        logger.info(f"Shape of db {data_db.shape}")
        n_gpus = faiss.get_num_gpus()
        logger.info(f"Number of GPUs available: {n_gpus}")
        index = faiss.IndexFlatL2(data_db.shape[1])
        if opts.get("mips", True):
            index = faiss.IndexIVFFlat(
                index, data_db.shape[1], opts.get("nlist", 100), faiss.METRIC_L2
            )
        if n_gpus > 0:
            logger.info("Building GPU model")
            index = faiss.index_cpu_to_all_gpus(index)

        logger.info("Builing Indexes")
        data_db = np.float32(data_db)
        if opts.get("mips", True):
            index.train(data_db)
        index.add(data_db)
        logger.info(f"Gpu Index: {index.ntotal}")
        return index


class FaissSearchTask(Task):
    @overrides
    def run(self, index, query_dict, k=5, opts={}):
        logger.info(f"Searching over Faiss")
        query_db = reduce(
            lambda data, val: np.vstack((data, val)) if data is not None else val,
            query_dict.values(),
            None,
        )
        query_db = np.float32(query_db)
        D, I = index.search(query_db, k)
        index.nprobe = opts.get("nprobe", 1)

        query_output = {
            id: {"distances": D[i], "index": I[i]} for i, id in enumerate(query_dict)
        }
        return query_output
