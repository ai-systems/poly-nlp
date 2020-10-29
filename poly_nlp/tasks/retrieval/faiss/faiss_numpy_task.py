from prefect import Task
from overrides import overrides
from typing import Dict
import numpy as np


class FaissIndexBuildTask(Task):
    @overrides
    def run(self, data: Dict[str, np.ndarray]):
        ...
