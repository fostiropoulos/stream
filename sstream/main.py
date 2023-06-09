import itertools
from pathlib import Path
from typing import List, Tuple, Type
from sstream.dataset.main import SurpriseDataset
import numpy as np
from torch.utils.data import DataLoader, Dataset

from sstream.dataset.scifar import SplitCIFAR100, SurpriseCIFAR
from sstream.dataset.smnist import PermutedMnist, SurpriseMNIST, SurpriseVectorMNIST
from sstream.dataset.sstream import SurpriseModal
from sstream.dataset.utils import (
    get_random_seed,
    make_dataloader,
)

import typing as ty


def get_task_class(
    name: ty.Literal["mnist", "pmnist", "mnistv", "cifar", "splitcifar", "smodal"]
):

    if name == "mnist":
        return SurpriseMNIST
    elif name == "pmnist":
        return PermutedMnist
    elif name == "mnistv":
        return SurpriseVectorMNIST
    elif name == "cifar":
        return SurpriseCIFAR
    elif name == "splitcifar":
        return SplitCIFAR100
    elif name == "smodal":
        return SurpriseModal
    else:
        raise NotImplementedError


class TaskScheduler:
    def __init__(
        self,
        dataset_root_path: Path,
        dataset: ty.Literal[
            "mnist", "pmnist", "mnistv", "cifar", "splitcifar", "smodal"
        ],
        batch_size: int,
        stream_seed: int = 0,
        num_rotations: int = 5,
        num_perm: int = 5,
    ):
        super().__init__()
        # task
        self.task_class: Type[SurpriseDataset] = get_task_class(dataset)
        self.ds_names = self.task_class.DS_NAMES
        self.seed = stream_seed
        self.batch_size = batch_size
        self.dataset_root_path = Path(dataset_root_path)
        # will fix model initialization as well
        self.num_rotations = num_rotations
        self.num_perm = num_perm
        self.task_names = self._create_tasks()
        self.task = 0
        self.n_tasks = len(self.task_names)
        self.total_steps = sum(
            [
                len(
                    self.task_class(
                        root=self.dataset_root_path,
                        task_name=task_name,
                        split="train",
                    )
                )
                for task_name in self.task_names
            ]
        )
        self.ds: Dataset
        self.dl: DataLoader

    def make_backbone_config(self):
        return self.task_class.make_backbone_config()

    def _make_task_names(self, n_perm, n_rot, is_vector=False):
        rng = np.random.RandomState(seed=self.seed)

        rots = []
        if n_rot > 0:
            limits = (
                [30 * i for i in range(12)]
                if not is_vector
                else [5 * i for i in range(1, 8)]
            )
            rots = rng.choice(limits, size=n_rot, replace=False).tolist()
        perms = []
        if n_perm > 0:
            perms = [get_random_seed() for i in range(n_perm)]

        # generate tasks
        perm_postfix = [f"perm_{seed}" for seed in perms]
        rot_postfix = [f"rot_{seed}" for seed in rots]
        post_fix = perm_postfix + rot_postfix
        # Cartesian product
        assert len(post_fix) > 0
        task_names = list(itertools.product(*(self.ds_names, post_fix)))
        task_names = [f"{x}_{y}" for x, y in task_names]

        return task_names

    def _create_tasks(self) -> List[str]:
        is_vector = len(self.task_class.DATA_SHAPE) == 1
        reg_tasks = self._make_task_names(
            n_perm=self.num_perm,
            n_rot=self.num_rotations,
            is_vector=is_vector,
        )

        return reg_tasks

    def __iter__(self):
        return self

    @property
    def task_name(self) -> str:
        return self.task_names[self.task]

    def __next__(self) -> Tuple[DataLoader, DataLoader]:
        if self.task >= self.n_tasks:
            raise StopIteration
        self.dl = self.get_dataloader(split="val", batch_size=self.batch_size)
        val_dl = self.get_dataloader(split="val", batch_size=self.batch_size)
        self.task += 1
        return self.dl, val_dl

    def __len__(self):
        return len(self.task_names)

    def get_dataloader(self, split: ty.Literal["train", "val"], batch_size: int):

        self.ds = self.task_class(
            root=self.dataset_root_path,
            task_name=self.task_name,
            split=split,
        )
        return make_dataloader(
            self.ds, batch_size=batch_size, split=split, sampler=None
        )
