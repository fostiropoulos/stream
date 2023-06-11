import argparse
from pathlib import Path

import torch
import sys
from autods.datasets import Amazon, DomainNetReal, DomainNetSketch, Imdb
from autods.main import AutoDS
from tqdm import tqdm

from stream.dataset import SurpriseDataset
from stream.dataset.transform.transforms import make_transform


class SurpriseModal(SurpriseDataset):
    """
    Using ImagenetR which is a rendition of Imagenet and Imagenet subsampled to
    200 classes present in ImagenetR
    https://arxiv.org/pdf/2204.04799.pdf
    https://arxiv.org/pdf/2006.16241.pdf

    Combining with Text Classification between Yelp and cross-domain shift
    to amazon reviews.

    Using ViT and GPT2 embeddings as a common representation between text and vision.
    """

    DATA_SHAPE = [
        768,
    ]
    HEAD_SIZE = 10
    DS = [DomainNetReal, DomainNetSketch, Imdb, Amazon]

    DS_TYPES = ["vision", "vision", "text", "text"]

    def __init__(self, root, task_name, split):
        super().__init__(root, task_name, split)

        self.dataset_args = dict(
            root_path=Path(root),
            task_id=0,
            feats_name="default",
            train=self.split == "train",
            datasets=[self.ds_name.lower()],
            transform=None,
        )
        self.transform_fn = self.transform()
        self.make_dataset()

    def make_dataset_class(self):
        return AutoDS

    @staticmethod
    def make_backbone_config():
        return dict(
            name="linear",
            input_dim=SurpriseModal.DATA_SHAPE,
            output_dim=SurpriseModal.HEAD_SIZE,
        )

    def __getitem__(self, idx):
        return_dict = super().__getitem__(idx)
        return_dict["x"] = self.transform_fn(return_dict["x"])
        return return_dict

    def transform(self):
        return make_transform(
            self.transform_name,
            "stream",
            self.val,
            split=self.split,
            is_vector=True,
        )


def _download_ds(ds, root_path):
    dataset = ds(root_path=root_path, action="download")


def download_dataset(
    root_path=Path.home().joinpath("stream_ds"),
    dataset_names=[Amazon, DomainNetReal, DomainNetSketch, Imdb],
):
    ps = []
    from multiprocessing import Process

    for ds in dataset_names:
        p = Process(target=_download_ds, args=(ds, root_path))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()


def extract_feats(
    root_path=Path.home().joinpath("stream_ds"),
    dataset_names=[Amazon, DomainNetReal, DomainNetSketch, Imdb],
    device: str = "cuda",
    batch_size: int = 64,
):
    for ds in dataset_names:
        dataset = ds(root_path=root_path)
        dataset.make_features(batch_size, device, feats_name=ds.default_feat_extractor)
        for train in [True, False]:
            dataset = ds(
                root_path=root_path,
                train=train,
                feats_name=ds.default_feat_extractor,
            )
            for s in tqdm(dataset):
                pass


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Download and process SStream Dataset")
    args.add_argument(
        "--action",
        required=True,
        choices=["download", "make"],
    )
    args.add_argument("--dataset_path", required=True, type=Path)
    if "--action make" in sys.argv:
        args.add_argument(
            "--device",
            required=True,
            choices=["cuda", "cpu"]
            + [f"cuda:{i}" for i in range(torch.cuda.device_count())],
        )
        args.add_argument("--batch_size", required=True, type=int)
    pargs = args.parse_args()

    # 1.
    if pargs.action == "download":
        download_dataset(pargs.dataset_path)
    # 2.
    elif pargs.action == "make":
        extract_feats(
            pargs.dataset_path, device=pargs.device, batch_size=pargs.batch_size
        )
