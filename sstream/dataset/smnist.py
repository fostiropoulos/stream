from torchvision.datasets import MNIST, SVHN

from sstream.dataset import SurpriseDataset
from sstream.dataset.transform.transforms import make_transform


class SurpriseMNIST(SurpriseDataset):
    DATA_SHAPE = [32, 32, 1]
    HEAD_SIZE = 10
    DS = [MNIST, SVHN]
    DS_TYPES = ["vision", "vision"]

    def __init__(self, root, task_name, split, **kwargs):
        super().__init__(root, task_name, split)

        if self.ds == MNIST:
            self.dataset_args["train"] = split == "train"
            self.dataset_args["download"] = not self.dataset_path.joinpath(
                f"train-images-idx3-ubyte.gz"
            ).exists()
        elif self.ds == SVHN:
            split = "test" if split != "train" else split
            self.dataset_args["split"] = split
            self.dataset_args["download"] = not self.dataset_path.joinpath(
                f"{split}_32x32.mat"
            ).exists()

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
            input_dim=SurpriseMNIST.DATA_SHAPE,
            output_dim=SurpriseMNIST.HEAD_SIZE,
        )


class SurpriseVectorMNIST(SurpriseMNIST):
    DATA_SHAPE = [
        28 * 28,
    ]

    def transform(self):
        return make_transform(
            self.transform_name,
            self.ds_name,
            self.val,
            split=self.split,
            is_vector=True,
        )

    @staticmethod
    def make_backbone_config():
        return dict(
            name="linear",
            input_dim=SurpriseVectorMNIST.DATA_SHAPE,
            output_dim=SurpriseVectorMNIST.HEAD_SIZE,
        )


class PermutedMnist(SurpriseVectorMNIST):
    DS = [MNIST]
