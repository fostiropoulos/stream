import os
from pathlib import Path
import numpy as np

from pyunpack import Archive

from torchvision.datasets import ImageFolder

from torchvision.datasets import CIFAR10, CIFAR100

from stream.dataset import SurpriseDataset
from stream.dataset.transform.transforms import make_transform


def download_cinic10(root: Path, extract_dir: Path):
    import kaggle

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files("mengcius/cinic10", path=root)
    save_path = root.joinpath("cinic10.zip")
    extract_dir.mkdir(exist_ok=True, parents=True)
    Archive(save_path).extractall(extract_dir)
    # rename valid to val
    os.rename(extract_dir.joinpath("valid"), extract_dir.joinpath("val"))


def is_valid_cimagenet_file(path):
    return "cifar" not in Path(path).stem


class CImagenet(ImageFolder):
    """CImagenet is a Cifar10 classmap of Imagenet from CINIC-10 dataset"""

    def __init__(self, root, split, download=False, **kwargs) -> None:
        self.data_dir = Path(root).joinpath("cinic-10")
        download = not self.data_dir.exists()
        assert (
            self.data_dir.exists() if not download else True
        ), "Data-dir does not exist."
        if download:
            download_cinic10(root, self.data_dir)
        self.image_dir = self.data_dir.joinpath(split)
        self.split = split
        super().__init__(
            self.image_dir, is_valid_file=is_valid_cimagenet_file, **kwargs
        )


class SurpriseVision(SurpriseDataset):
    DATA_SHAPE = [3, 32, 32]
    HEAD_SIZE = 10
    DS = [CIFAR10, CImagenet]
    DS_TYPES = ["vision", "vision"]

    def __init__(self, root, task_name, split, **kwargs):
        super().__init__(root, task_name, split)

        if self.ds == CIFAR10:
            self.dataset_args["train"] = split == "train"
        elif self.ds == CImagenet:
            self.dataset_args["split"] = split

        self.make_dataset()

    def transform(self):
        return make_transform(
            self.transform_name,
            self.ds_name,
            self.val,
            split=self.split,
            is_vector=False,
        )

    @staticmethod
    def make_backbone_config():
        return dict(
            name="resnet18",
            input_dim=SurpriseVision.DATA_SHAPE,
            output_dim=SurpriseVision.HEAD_SIZE,
        )


class SplitCIFAR100(SurpriseDataset):
    DATA_SHAPE = [3, 32, 32]
    HEAD_SIZE = 10
    DS = [CIFAR100]
    DS_TYPES = ["vision"]

    def __init__(self, root, task_name, split, **kwargs):
        super().__init__(root, task_name, split)

        self.dataset_args["train"] = split == "train"
        self.make_dataset()

        self.split_id = int(self.ds_fullname.split("-")[1])

        classes = np.arange(100).reshape(-1, 10)

        mask = [t in classes[self.split_id] for t in self.dataset.targets]
        self.idxs = np.argwhere(mask).flatten()

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        x, y = self.dataset[idx]
        y = y % 10
        return {"x": x, "y": y}

    def __len__(self):
        return len(self.idxs)

    def transform(self):
        return make_transform(
            "rot",
            self.ds_name,
            0,
            split=self.split,
            is_vector=False,
        )

    def make_dataset_class(self):
        return CIFAR100

    @staticmethod
    def make_backbone_config():
        return dict(
            name="resnet18",
            input_dim=SurpriseVision.DATA_SHAPE,
            output_dim=SurpriseVision.HEAD_SIZE,
        )

    @classmethod
    @property
    def DS_NAMES(cls):
        return [f"vision:CIFAR100-{i}" for i in range(10)]
