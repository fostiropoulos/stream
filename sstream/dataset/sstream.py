from pathlib import Path

from stream.datasets import Amazon, DomainNetReal, DomainNetSketch, Imdb
from stream.main import Stream
from tqdm import tqdm

from sstream.dataset import SurpriseDataset
from sstream.dataset.transform.transforms import make_transform


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
        return Stream

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
        # dataset = ds(root_path=root_path, make=True)
        # dataset = ds(root_path=root_path, action="download")
        p = Process(target=_download_ds, args=(ds, root_path))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()


def extract_feats(
    root_path=Path.home().joinpath("stream_ds"),
    dataset_names=[Amazon, DomainNetReal, DomainNetSketch, Imdb],
):
    for ds in dataset_names:
        dataset = ds(root_path=root_path)
        dataset.make_features(64, "cuda", feats_name=ds.default_feat_extractor)
        for train in [True, False]:
            dataset = ds(
                root_path=root_path,
                train=train,
                feats_name=ds.default_feat_extractor,
            )
            for s in tqdm(dataset):
                pass


def export_feats(
    root_path=Path.home().joinpath("stream_ds"),
    dataset_names=["amazon", "domainnetreal", "domainnetsketch", "imdb"],
    export_path=Path.home().joinpath("surprise_feats"),
):
    for train in [True, False]:
        for task_id in range(len(dataset_names)):
            dataset = Stream(
                root_path,
                datasets=dataset_names,
                task_id=task_id,
                feats_name="default",
                train=[train],
            )
            for t in tqdm(dataset):
                pass
    dataset.export_feats(export_path, export_all=False, clean_make=True)


if __name__ == "__main__":
    # 1.
    # download_dataset()
    # 2.
    extract_feats()
    # 3.
    export_feats()
