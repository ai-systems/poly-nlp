import math
import os
from functools import reduce
from typing import Dict

import numpy as np
from loguru import logger
from overrides import overrides
from prefect import Task

import faiss


class FaissIndexBuildTask(Task):
    @overrides
    def run(self, data: Dict[str, np.ndarray], faiss_task, output_dir, opts={}):
        save_path = f"{output_dir}/{faiss_task}.in"
        if os.path.isfile(save_path):
            logger.info(f"Reading from saved path: {save_path}")
            index = faiss.read_index(save_path)
            index = faiss.index_cpu_to_all_gpus(index)
        else:
            logger.info(f"Constructing Faiss: {opts}")
            # data_db = reduce(
            #     lambda data, val: np.vstack((data, val)) if data is not None else val,
            #     data.values(),
            #     None,
            # )
            data_db = np.array(list(data.values()))
            logger.info(f"Shape of db {data_db.shape}")
            n_gpus = faiss.get_num_gpus()
            logger.info(f"Number of GPUs available: {n_gpus}")
            if opts.get("mips", True):
                logger.info("Building MIPS indexes")
                index = faiss.IndexFlatIP(data_db.shape[1])
                index = faiss.IndexIVFFlat(
                    index,
                    data_db.shape[1],
                    opts.get("nlist", int(2 * math.sqrt(len(data)))),
                    faiss.METRIC_INNER_PRODUCT,
                )
            else:
                index = faiss.IndexFlatL2(data_db.shape[1])
            if n_gpus > 0:
                logger.info("Building GPU model")
                index = faiss.index_cpu_to_all_gpus(index)

            logger.info("Builing Indexes")
            data_db = np.float32(data_db)
            if opts.get("mips", True):
                index.train(data_db)
            index.add(data_db)
            logger.info(f"Gpu Index: {index.ntotal}")
            faiss.write_index(faiss.index_gpu_to_cpu(index), save_path)
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
        index.nprobe = opts.get("nprobe", 4)

        query_output = {
            id: {I[i, pos]: D[i, pos] for pos in enumerate(I)}
            for i, id in enumerate(query_dict)
        }
        return query_output
