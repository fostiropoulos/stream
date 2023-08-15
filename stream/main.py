import itertools
from pathlib import Path
from typing import List, Tuple, Type
from stream.dataset.main import SurpriseDataset
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from stream.dataset.svis import SplitCIFAR100, SurpriseVision
from stream.dataset.snum import PermutedMnist, SurpriseNum, SurpriseVectorNum
from stream.dataset.smodal import SurpriseModal
from stream.dataset.utils import (
    get_random_seed,
    make_dataloader,
)

import typing as ty


task_class_map: dict[str, Type] = {
    "snum": SurpriseNum,
    "pmnist": PermutedMnist,
    "snumv": SurpriseVectorNum,
    "svis": SurpriseVision,
    "splitcifar": SplitCIFAR100,
    "smodal": SurpriseModal,
}


class TaskScheduler:
    """
    A TaskScheduler that create a sequence of tasks for a given number of permutation and
    rotation transformations.

    Attributes
    ----------
    task_class : Type[SurpriseDataset]
        The class to use to instatiate the dataset of each task.
    ds_names : list[str]
        The names of the sequence of datasets.
    seed : int
        The random seed for the dataset.
    batch_size : int
        The batch size used to instatiate the DataLoader
    dataset_root_path : Path
        The location of the dataset directory
    num_rotations : int
        The number of rotations that are added on the task-sequence
    num_perm : int
        The number of permutations that are added on the task-sequence
    task_names : list[str]
        The task names
    task : str
        The task
    n_tasks : int
        The number of tasks.
    total_steps : int
        The total number of steps.
    ds : Dataset
        The dataset class
    dl : DataLoader
        The DataLoader

    Parameters
    ----------
    dataset_root_path : Path | str
        the rootpath where the dataset files are stored. e.g. you can download the preprocessed features from
        `https://drive.google.com/file/d/1EYXOo4xEXSLwl2bim4BE9EiR4Km4HOkQ/view?usp=sharing`

    dataset : ty.Literal["snum", "pmnist", "snumv", "svis", "splitcifar", "smodal"]
        The name of the Stream sequence to instantiate
    batch_size : int
        The batch size for the task
    stream_seed : int, optional
        the seed to use for creating the transformations, by default 0
    num_rotations : int, optional
        the number of rotations to apply on each task, by default 5
    num_perm : int, optional
        the number of permutations to apply on each task, by default 5

    Examples
    --------
    Using TaskScheduler with "snumv" task sequence.

    >>> surprise_stream = TaskScheduler(
    ...    dataset="snumv", dataset_root_path=dataset_path, batch_size=128
    ... )
    >>> train_task, val_task = next(iter(surprise_stream))
    >>> train_batch = next(iter(train_task))
    >>> train_batch["x"].shape
    torch.Size([128, 1, 28, 28])
    >>> train_batch["y"].shape
    torch.Size([128])
    """

    def __init__(
        self,
        dataset_root_path: Path | str,
        dataset: ty.Literal["snum", "pmnist", "snumv", "svis", "splitcifar", "smodal"],
        batch_size: int,
        stream_seed: int = 0,
        num_rotations: int = 5,
        num_perm: int = 5,
    ):
        super().__init__()
        # task
        self.task_class: Type[SurpriseDataset] = task_class_map[dataset]
        self.ds_names: list[str] = self.task_class.DS_NAMES
        self.seed: int = stream_seed
        self.batch_size: int = batch_size
        self.dataset_root_path: Path = Path(dataset_root_path)
        # will fix model initialization as well
        self.num_rotations: int = num_rotations
        self.num_perm: int = num_perm
        self.task_names: list[str] = self._create_tasks()
        self.task: int = 0
        self.n_tasks: int = len(self.task_names)
        self.total_steps: int = sum(
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

    def make_backbone_config(self) -> dict[str, ty.Any]:
        """
        Returns the configuration of the backbone. The configuration
        should contain sufficient information to instatiate the model.
        The specific configuration depends on the ``SurpriseDataset``
        class. e.g. ``SurpriseVision``, ``SurpriseNum``,
        ``SurpriseVectorNum``, ``SurpriseModal``


        Returns
        -------
        dict[str, ty.Any]
            the backbone configuration as a dictionary with keys:
            "name", "input_dim", "output_dim"

        Examples
        --------
        Using TaskScheduler with "snumv" task sequence.

        >>> surprise_stream = TaskScheduler(
        ...    dataset="snumv", dataset_root_path=dataset_path, batch_size=128
        ... )
        >>> surprise_stream.make_backbone_config()
        {'name': 'linear', 'input_dim': [784], 'output_dim': 10}
        """
        return self.task_class.make_backbone_config()

    def _make_task_names(self, n_perm, n_rot, is_vector=False):
        rng = np.random.RandomState(seed=self.seed)
        generator = torch.Generator().manual_seed(self.seed)

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
            perms = [get_random_seed(generator) for i in range(n_perm)]

        # generate tasks
        perm_postfix = [f"perm_{seed}" for seed in perms]
        rot_postfix = [f"rot_{seed}" for seed in rots]
        post_fix = perm_postfix + rot_postfix
        # Cartesian product
        assert len(post_fix) > 0
        task_names = list(itertools.product(*(self.ds_names, post_fix)))
        task_names = [f"{x}_{y}" for x, y in task_names]
        rng.shuffle(task_names)
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
        """
        The current task for the Stream sequence can change.
        This property returns the name of the current task.

        Returns
        -------
        str
            the current task name.

        Examples
        --------
        Using TaskScheduler with "snumv" task sequence.

        >>> surprise_stream = TaskScheduler(
        ...    dataset="snumv", dataset_root_path=dataset_path, batch_size=128
        ... )
        >>> surprise_stream.task_name
        'vision:MNIST_perm_924231285'
        """
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

    def get_dataloader(
        self, split: ty.Literal["train", "val"], batch_size: int
    ) -> DataLoader:
        """
        The current task for the Stream sequence can change. This method
        can be used to return the dataloader for the current task

        Returns
        -------
        str
            the current task name.

        Examples
        --------
        Using TaskScheduler with "snumv" task sequence.

        >>> surprise_stream = TaskScheduler(
        ...    dataset="snumv", dataset_root_path=dataset_path, batch_size=128
        ... )
        >>> dataloader = surprise_stream.get_dataloader(split="train", batch_size=128)
        >>> batch = next(iter(dataloader))
        >>> batch["x"].shape
        torch.Size([128, 1, 28, 28])


        Parameters
        ----------
        split : ty.Literal["train", "val"]
            the dataset split for which to return the DataLoader
        batch_size : int
            the batch-size by which to instantiate the DataLoader

        Returns
        -------
        DataLoader
            The DataLoader corresponding to the current task.
        """

        self.ds = self.task_class(
            root=self.dataset_root_path,
            task_name=self.task_name,
            split=split,
        )
        return make_dataloader(
            self.ds, batch_size=batch_size, split=split, sampler=None
        )
