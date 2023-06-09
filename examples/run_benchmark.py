import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sstream import TaskScheduler
from sstream.modules.backbone import Backbone, make_backbone
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import typing as ty


@torch.no_grad()
def val(backbone: Backbone, val_tasks: list[DataLoader]):
    backbone.eval()
    evals: list[float] = []
    for val_task in val_tasks:
        predictions = []
        targets = []
        for batch in val_task:
            x = batch["x"]
            y = batch["y"]

            logits: torch.Tensor = backbone(x, compute_feats=True)
            predictions.append(logits)
            targets.append(y)
        predictions = torch.concat(predictions)
        targets = torch.concat(targets)
        evals.append(
            roc_auc_score(
                targets,
                predictions.softmax(-1),
                multi_class="ovo",
            )
        )
    backbone.train()
    return evals


def train(backbone: Backbone, train_task: DataLoader):
    optimizer = optim.SGD(backbone.parameters(), lr=0.01, momentum=0.9)
    backbone.train()

    t = tqdm(train_task)
    task_name = train_task.dataset.task_name
    for batch in t:
        x = batch["x"]
        y = batch["y"]
        optimizer.zero_grad()
        # NOTE you can apply your GCL method here
        logits = backbone(x, compute_feats=True)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        t.set_description(f"Task {task_name} loss:{loss.item():.4f}")
        optimizer.step()


def run(
    dataset_name: ty.Literal[
        "snum", "pmnist", "snumv", "svis", "splitcifar", "smodal"
    ],
    dataset_path: Path,
):
    surprise_stream = TaskScheduler(
        dataset=dataset_name, dataset_root_path=dataset_path, batch_size=128
    )
    config = surprise_stream.make_backbone_config()
    # TODO make me (backbone) not forget
    dummy_learner: Backbone = make_backbone(**config)
    val_tasks = []
    # Stores predictions of each previous task after having been trained
    # on a new task
    r_matrix = []
    for train_task, val_task in surprise_stream:
        train(dummy_learner, train_task)
        val_tasks.append(val_task)
        r_matrix.append(val(dummy_learner, val_tasks))
        score = np.mean(r_matrix[-1])
        print(f"Current Score: {score:.4f}")

    print(f"Final Score: {score:.4f}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(
        description="Train a Dummy Learner on Surprise Stream"
    )

    args.add_argument(
        "--dataset_name",
        required=True,
        choices=["snum", "pmnist", "snumv", "svis", "splitcifar", "smodal"],
    )
    args.add_argument("--dataset_path", required=True, type=Path)

    kwargs = vars(args.parse_args())
    run(**kwargs)
