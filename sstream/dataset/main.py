from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
import typing as ty

from torch.utils.data import Dataset


class SurpriseDataset(Dataset, ABC):
    def __init__(self, root: str, task_name: str, split: str):
        super().__init__()
        self.split = split
        self.task_name = task_name
        self.ds_fullname, self.transform_name, seed = task_name.split("_")
        self.val = int(seed)
        # dataset
        self.ds_type, self.ds_name = self.ds_fullname.split(":")
        self.ds_names = [getattr(ds, "__name__", ds) for ds in self.DS]
        self.ds = self.make_dataset_class()

        self.dataset_path = Path(root).joinpath(self.ds_name)
        self.dataset_args = {
            "root": self.dataset_path,
            "download": not self.dataset_path.exists(),
            "transform": self.transform(),
        }

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return {"x": x, "y": y}

    def __len__(self):
        return len(self.dataset)

    def make_dataset_class(self):
        ds_idx = self.ds_names.index(self.ds_name)
        ds = self.DS[ds_idx]
        return ds

    def make_dataset(self):
        self.dataset = self.ds(**self.dataset_args)

    @abstractmethod
    def transform(self):
        pass

    @staticmethod
    @abstractmethod
    def make_backbone_config():
        pass

    @abstractproperty
    def DATA_SHAPE(self):
        pass

    @abstractproperty
    def HEAD_SIZE(self):
        pass

    @abstractproperty
    def DS(self) -> list[type[Dataset]]:
        pass

    @abstractproperty
    def DS_TYPES(self) -> list[str]:
        pass

    @classmethod
    @property
    def DS_NAMES(cls):
        return [
            ds_type + ":" + getattr(ds, "__name__", ds)
            for ds_type, ds in zip(cls.DS_TYPES, cls.DS)
        ]
