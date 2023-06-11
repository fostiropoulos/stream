from typing import Callable, List
from torchvision.transforms import transforms

from stream.dataset.transform.permutation import (
    FixedPermutation,
    FixedChannelPermutation,
)
from stream.dataset.transform.rotation import (
    FixedVectorRotation,
    FixedChannelRotation,
)


def make_ds_transform(ds_name: str, split: str, is_vector=False) -> List[Callable]:
    if ds_name == "mnist" and is_vector:
        base_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    elif ds_name == "svhn" and is_vector:

        base_transform = [
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]

    elif ds_name == "mnist" and not is_vector:

        base_transform = [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    elif ds_name == "svhn" and not is_vector:
        base_transform = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    elif ds_name in {"cifar10", "cifar100"} or ds_name.startswith("cifar100"):
        if split != "train":
            base_transform = [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        else:
            base_transform = [
                transforms.Resize(32),
                # transforms.RandomCrop(64, padding=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
    elif ds_name == "cimagenet":
        if split != "train":
            base_transform = [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        else:
            base_transform = [
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]

    elif ds_name == "stream":
        base_transform = []

    else:
        raise NotImplementedError
    return base_transform


def make_task_transform(transform_name: str, val: int) -> List[Callable]:
    task_transform: List[Callable]
    if transform_name == "rot":
        task_transform = [FixedChannelRotation(deg=val)]
    elif transform_name == "perm":
        task_transform = [FixedChannelPermutation(seed=val)]
    else:
        raise NotImplementedError
    return task_transform


def make_transform(transform_name, ds_name, val, split, is_vector=False):
    ds_name = ds_name.lower()
    ds_transform = make_ds_transform(ds_name, split, is_vector=is_vector)
    if transform_name is None:
        task_transform = []
    elif is_vector:
        task_transform = make_vector_transform(transform_name, val)
    else:
        task_transform = make_task_transform(transform_name, val)
    return transforms.Compose(ds_transform + task_transform)


def make_vector_transform(transform_name: str, val: int):
    if transform_name == "perm":
        return [FixedPermutation(seed=val)]
    elif transform_name == "rot":
        return [FixedVectorRotation(deg=val)]
    else:
        raise NotImplementedError
