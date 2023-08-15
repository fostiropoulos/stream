from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
import typing as ty
import torch

from torch.utils.data import Dataset


class SurpriseDataset(Dataset, ABC):
    """
    The Base Class for the SurpriseDataset.
    It should implement a list of datasets $D_\text{base}$
    on which the transformations will be applied.

    Attributes
    ----------
    split : str
        The train/test split to use. It will be provided as argument to the Dataset
        constructor
    task_name : str
        The task name in the compressed format.
    ds_fullname : str
        The dataset full name, e.g. 'vision:SVHN'
    transform_name : str
        The transformation name, e.g. 'perm'
    seed : int
        The seed used for the transformation, e.g. '398764591'
    ds_type : str
        The dataset type e.g. 'vision'
    ds_name : str
        The dataset name e.g. 'SVHN'
    ds_names : list[str]
        The dataset names of all available tasks e.g. `['MNIST', 'SVHN']`
    ds : Dataset
        The dataset class of the current task.
    dataset_path : Path
        The path of where the dataset is located.
    dataset_args : dict[str, ty.Any]
        The arguments of the dataset constructor.



    Examples
    --------
    A SurpriseDataset composed of MNIST and SVHN
    >>> from torchvision.datasets import MNIST, SVHN
    >>> from stream.dataset.transform.transforms import make_transform
    >>> class SurpriseNum(SurpriseDataset):
    ...    DATA_SHAPE = [32, 32, 1]
    ...    HEAD_SIZE = 10
    ...    DS = [MNIST, SVHN]
    ...    DS_TYPES = ["vision", "vision"]
    ...    def __init__(self, root, task_name, split, **kwargs):
    ...        super().__init__(root, task_name, split)
    ...        if self.ds == MNIST:
    ...            self.dataset_args["train"] = split == "train"
    ...            self.dataset_args["download"] = not self.dataset_path.joinpath(
    ...                f"train-images-idx3-ubyte.gz"
    ...            ).exists()
    ...        elif self.ds == SVHN:
    ...            split = "test" if split != "train" else split
    ...            self.dataset_args["split"] = split
    ...            self.dataset_args["download"] = not self.dataset_path.joinpath(
    ...                f"{split}_32x32.mat"
    ...            ).exists()
    ...        self.make_dataset()
    ...    def transform(self):
    ...        return make_transform(
    ...            self.transform_name,
    ...            self.ds_name,
    ...            self.seed,
    ...            split=self.split,
    ...            is_vector=False,
    ...        )
    ...    @staticmethod
    ...    def make_backbone_config():
    ...        return dict(
    ...            name="resnet18",
    ...            input_dim=SurpriseNum.DATA_SHAPE,
    ...            output_dim=SurpriseNum.HEAD_SIZE,
    ...        )
    >>> # Dataset Type, Dataset name, Transformation, seed
    >>> task_name = 'vision:SVHN_perm_398764591'
    >>> dataset = SurpriseNum("/tmp/snum", task_name)
    >>> # We pass through a frozen ResNet18 backbone first.
    >>> dataset[0]["x"].shape
    torch.Size([64, 16, 16])
    """

    def __init__(self, root: Path | str, task_name: str, split: str):
        super().__init__()
        self.split: str = split
        self.task_name: str = task_name
        self.ds_fullname, self.transform_name, seed = task_name.split("_")
        self.seed: int = int(seed)
        # dataset
        self.ds_type, self.ds_name = self.ds_fullname.split(":")
        self.ds_names = [getattr(ds, "__name__", ds) for ds in self.DS]
        self.ds = self.make_dataset_class()

        self.dataset_path: Path = Path(root).joinpath(self.ds_name)
        self.dataset_args: dict[str, ty.Any] = {
            "root": self.dataset_path,
            "download": not self.dataset_path.exists(),
            "transform": self.transform(),
        }

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        x, y = self.dataset[idx]
        return {"x": x, "y": y}

    def __len__(self) -> int:
        return len(self.dataset)

    def make_dataset_class(self):
        ds_idx = self.ds_names.index(self.ds_name)
        ds = self.DS[ds_idx]
        return ds

    def make_dataset(self):
        self.dataset = self.ds(**self.dataset_args)

    @abstractmethod
    def transform(self) -> ty.Callable:
        ...

    @staticmethod
    @abstractmethod
    def make_backbone_config() -> dict[str, ty.Any]:
        ...

    @abstractproperty
    def DATA_SHAPE(self) -> tuple[int, ...]:
        ...

    @abstractproperty
    def HEAD_SIZE(self) -> int:
        ...

    @abstractproperty
    def DS(self) -> list[type[Dataset]]:
        ...

    @abstractproperty
    def DS_TYPES(self) -> list[str]:
        ...

    @classmethod
    @property
    def DS_NAMES(cls) -> list[str]:
        return [
            ds_type + ":" + getattr(ds, "__name__", ds)
            for ds_type, ds in zip(cls.DS_TYPES, cls.DS)
        ]
