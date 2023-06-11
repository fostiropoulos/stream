import torch
import torch
from torch.utils.data import DataLoader


def get_generator(seed=None):
    if seed is None:
        seed = get_random_seed()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_random_seed(generator=None):
    return int(torch.empty((), dtype=torch.int32).random_(generator=generator).item())


def make_dataloader(dataset, batch_size, split="train", sampler=None):
    if sampler is not None:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=split == "train",
            num_workers=1,
            prefetch_factor=5,
            persistent_workers=True
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            drop_last=split == "train",
            num_workers=1,
            prefetch_factor=5,
            persistent_workers=True
        )
    return dataloader
