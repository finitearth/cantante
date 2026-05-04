"""
Functions for loading and preparing datasets based on configuration parameters.
Handles retrieving datasets from Hugging Face, applying preprocessing,
and formatting them according to task requirements.
"""

from typing import Literal

from datasets import load_dataset

from src.experiment.configs import ALL_DATASETS
from src.tasks.base import BaseMASTask
from src.tasks.gsm8k import GSM8KTask
from src.tasks.hotpotqa import HotpotQATask
from src.tasks.mbpp import MBPPTask


def get_tasks(
    dataset_name: str,
    split: Literal["train", "test"],
    block_size: int = 30,
    dev_size: int = 300,
    test_size: int = 500,
    seed: int = 42,
    eval_strategy: Literal[
        "full", "subsample", "sequential_block", "random_block"
    ] = "sequential_block",
    tools_adapter=None,
    cache_dir: str = "data_cache",
) -> BaseMASTask:
    """
    Load one split of a dataset and return the corresponding task.

    Args:
        dataset_name: Name of the dataset (must be defined in ALL_DATASETS)
        split: Which split to load — "train" (dev) or "test"
        dev_size: Number of examples to sample for the train/dev split
        test_size: Number of examples to sample for the test split
        seed: Random seed for reproducibility
        cache_dir: Directory to cache downloaded datasets (gitignored by default)

    Returns:
        A single BaseMASTask for the requested split
    """
    config = ALL_DATASETS[dataset_name]

    hf_split = config.splits.train if split == "train" else config.splits.test
    hf_name = config.names.train if split == "train" else config.names.test
    size = dev_size if split == "train" else test_size
    df = load_dataset(
        config.name,
        name=hf_name,
        split=hf_split,
        revision=config.revision,
        cache_dir=cache_dir,
    ).to_pandas()  # type: ignore[union-attr]
    df = df.sample(size, random_state=seed, replace=False)  # type: ignore[union-attr]

    if callable(config.input):
        df.loc[:, "input"] = config.input(df)
    else:
        df.loc[:, "input"] = df[config.input]

    if callable(config.target):
        df.loc[:, "target"] = config.target(df)
    elif config.target == "":
        df.loc[:, "target"] = ""
    else:
        df.loc[:, "target"] = df[config.target]

    is_train = split == "train"

    if dataset_name == "mbpp":
        task = MBPPTask(
            df=df,
            x_column="input",
            y_column="target",
            **({"n_subsamples": block_size, "eval_strategy": eval_strategy} if is_train else {}),
        )

    elif dataset_name == "hotpotqa":
        task = HotpotQATask(
            df=df,
            x_column="input",
            y_column="target",
            **({"n_subsamples": block_size, "eval_strategy": eval_strategy} if is_train else {}),
        )
    elif dataset_name == "mbpp":
        task = MBPPTask(
            df=df,
            x_column="input",
            y_column="target",
            **({"n_subsamples": block_size, "eval_strategy": eval_strategy} if is_train else {}),
        )
    else:
        task = GSM8KTask(
            df=df,
            x_column="input",
            y_column="target",
            **({"n_subsamples": block_size, "eval_strategy": eval_strategy} if is_train else {}),
        )

    task.set_tools_adapter(tools_adapter)  # type: ignore
    return task
