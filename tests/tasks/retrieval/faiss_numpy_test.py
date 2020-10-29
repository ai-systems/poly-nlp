import unittest

import numpy as np
from poly_nlp.tasks.retrieval.faiss import FaissIndexBuildTask, FaissSearchTask


class FaissNumpyTest(unittest.TestCase):
    def test_faiss_build(self):
        # Note: need to fine tune n_list and nprobe
        index_db = {i: np.random.rand(768) for i in range(100)}
        query_db = {i: np.random.rand(768) for i in range(10)}
        index_model = FaissIndexBuildTask().run(index_db)
        query_ouput = FaissSearchTask().run(index_model, query_db, k=2)
        print(query_ouput)
