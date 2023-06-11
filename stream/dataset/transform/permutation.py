import numpy as np
import torch
from torchvision.models import resnet18

from stream.dataset.utils import get_generator


class FixedPermutation:
    """
    Defines a fixed permutation (given the seed) for a numpy array.
    """

    def __init__(self, seed: int) -> None:
        self.perm = None
        self.perm_column = None
        self.generator = get_generator(seed)

    def __call__(self, sample, **kwargs):
        old_shape = sample.shape
        sample = sample.flatten()
        if self.perm is None:
            self.perm = torch.randperm(sample.shape[0], generator=self.generator)
        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample)
        return sample[self.perm].reshape(old_shape)


class FixedChannelPermutation:
    """
    Defines a fixed permutation (given the seed) for a 64-channel array after 1st conv of pretrained resnet-18.
    """

    def __init__(self, seed: int) -> None:
        pretrained = resnet18(weights="IMAGENET1K_V1")
        self.conv = pretrained.conv1
        self.conv.eval()
        self.generator = get_generator(seed)
        self.perm = torch.randperm(self.conv.out_channels, generator=self.generator)

    @torch.no_grad()
    def __call__(self, sample, **kwargs):
        sample = sample.unsqueeze(0)
        sample_feat = self.conv(sample)
        return sample_feat.squeeze(0)[self.perm, :, :]
