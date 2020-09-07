from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class PolyTorchDataset(Dataset, ABC):
    @abstractmethod
    def get_id(self, index):
        ...
