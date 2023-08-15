from typing import Optional
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from torchvision.models import resnet18

from stream.dataset.utils import get_generator


class FixedRotation(object):
    """
    Defines a fixed rotation for a numpy array.
    """

    def __init__(
        self,
        deg: Optional[int] = None,
        seed: Optional[int] = None,
        deg_min: int = 0,
        deg_max: int = 180,
    ) -> None:
        if deg is None:
            assert (
                seed is not None
            ), "cannot set degree and random seed at the same time"
            self.deg_min = deg_min
            self.deg_max = deg_max

            self.generator = get_generator(seed)
            self.degrees = torch.randint(
                low=deg_min, high=deg_max, size=(1,), generator=self.generator
            ).item()
            # reverse
            if torch.rand(size=(1,)).item() > 0.5:
                self.degrees *= -1
        else:
            assert seed is None, "cannot set degree and random seed at the same time"
            self.degrees = deg

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return F.rotate(x, self.degrees)


class IncrementalRotation(object):
    """
    Defines an incremental rotation for a numpy array.
    """

    def __init__(
        self, init_deg: int = 0, increase_per_iteration: float = 0.006
    ) -> None:
        self.increase_per_iteration = increase_per_iteration
        self.iteration = 0
        self.degrees = init_deg

    def __call__(self, x: np.ndarray) -> np.ndarray:
        degs = (self.iteration * self.increase_per_iteration + self.degrees) % 360
        self.iteration += 1
        return F.rotate(x, degs)

    def set_iteration(self, x: int) -> None:
        self.iteration = x



class FixedVectorRotation(object):
    """
    Defines a fixed rotation for a numpy vector.

    Readings:

    https://math.stackexchange.com/questions/3698915/n-dimensional-rotation-matrix
    http://wscg.zcu.cz/wscg2004/Papers_2004_Short/N29.pdf
    "ChronoR: Rotation Based Temporal Knowledge Graph Embedding"
    """

    def __init__(self, deg: int) -> None:
        """
        Initializes the rotation with a random orthogonal matrix.
        """

        self.rot = deg


    def __call__(self, sample):
        old_shape = sample.shape
        flat_sample = sample.reshape(-1)
        n_features = flat_sample.shape[-1]
        image_size = int(np.ceil(n_features ** (0.5)))
        padding = int(image_size ** 2 - n_features)
        padded_sample = np.pad(flat_sample, (0, padding)).reshape(
            1, image_size, image_size
        )

        if isinstance(padded_sample, np.ndarray):
            padded_sample = torch.from_numpy(padded_sample)
        rotated_sample = F.rotate(padded_sample, self.rot).reshape(1,-1)[:,:n_features].reshape(old_shape)
        return rotated_sample


class FixedChannelRotation(FixedRotation):
    """
    Defines a fixed rotation for a 64-channel array after 1st conv of pretrained resnet-18.
    """

    def __init__(
        self,
        deg: Optional[int] = None,
        seed: Optional[int] = None,
        deg_min: int = 0,
        deg_max: int = 180,
    ) -> None:
        super().__init__(deg, seed, deg_min, deg_max)
        pretrained = resnet18(weights="IMAGENET1K_V1")
        self.conv = pretrained.conv1
        self.conv.eval()

    @torch.no_grad()
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        rotates an image in the feature space of a ResNet18.

        Parameters
        ----------
        x : np.ndarray
            image to be rotated

        Returns
        -------
        np.ndarray
            rotated image
        """
        x = x.unsqueeze(0)
        sample_feat = self.conv(x)
        return super().__call__(sample_feat.squeeze(0))

