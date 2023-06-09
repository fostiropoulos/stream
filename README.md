# Surprise-Stream

In  General Continual Learning (GCL), the goal is to learn a sequence of tasks that are presented once while maintaining performance on all previously learned tasks without the task identity during both the training and the evaluation phase. Stream provides a method to construct an infinite long sequence of tasks with varying degree of domain-gap (`learning-gap`) from a limited set of multi-modal dataset.

For example, one can use SVHN and MNIST (`snum`) to construct a sequence of tasks with different degree of `learning-gaps`.

## Usage

### Overview

1. Install
   We recommend using a virtual enviroment with Python>3.10
   ```bash
   git clone xxx
   cd xxx
   pip install  .
    ```
2. [Download the pre-processed dataset](https://drive.google.com/file/d/1EYXOo4xEXSLwl2bim4BE9EiR4Km4HOkQ/view?usp=sharing)

3. Modify [Example](examples/run_benchmark.py) to apply a GCL to avoid catastrophic forgetting


### In Detail:

1. After you download the dataset extract it to your desired location. The directory should have a structure like:

```bash
surprise_feats
    - amazon
        - feats
        . metadata.pickle
    - domainnetreal
    - domainnetsketch
    - imdb
```

2. You can run the dummy learner from the example by running the following command from the terminal
```
cd xxx
python examples/run_benchmark.py --dataset_name pmnist --dataset_path [save_directory]
```

3. In the example a dummy learner is provided by default as the `backbone` the goal is to make the `dummy_learner` not forget. Forgeting is evaluated as the auc score on all tasks learned so far. You can adapt the example to evaluate an existing method or design your own.
```python
# TODO make me (backbone) not forget
dummy_learner: Backbone = make_backbone(**config)
```
**NOTE** For a Generalized Continual Learning (GCL) setting:
1. The train tasks must appear once
2. You should not use information on the transitions between tasks (e.g. when a task ends and another begins) or their identity during both training and evaluation.
3. There should be a restriction on the resources when a new task arrives. Since the setting is for an infinite stream of tasks, it is trivial to create a copy of the same model to avoid forgetting or store all data as they come in. However the method should be designed such that it does not require an infinite amount of resources.


## Datasets


```python
from sstream import TaskScheduler
# You can set the dataset to be, ["snum", "pmnist", "snumv", "svis", "splitcifar", "smodal"]
surprise_stream = TaskScheduler(
        dataset="snumv", dataset_root_path=dataset_path, batch_size=128
    )
```

### GCL Datasets


1. SNum and SNum - Vector

[Description]

2. SVis

[Description]

3. SStream

[Description]

### CL Datasets

1. Split-Cifar100
[Description]
2. Permuted Mnist
[Description]
