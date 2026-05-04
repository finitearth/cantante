import re
from dataclasses import dataclass, field
from typing import Callable, List


@dataclass
class SplitConfig:
    train: str
    test: str


@dataclass
class DatasetConfig:
    name: str
    alias: str
    revision: str
    input: str | Callable
    target: str | Callable
    reward_columns: List[str] = field(default_factory=list)
    reward_function: Callable | None = None
    names: SplitConfig = field(default_factory=lambda: SplitConfig(train=None, test=None))
    splits: SplitConfig = field(default_factory=lambda: SplitConfig(train="train", test="test"))
    prebuild_instance_images: bool = True
    prebuild_max_workers: int = 4


_GSM8K_CONFIG = DatasetConfig(
    name="openai/gsm8k",
    alias="gsm8k",
    revision="e53f048856ff4f594e959d75785d2c2d37b678ee",
    input="question",
    target=lambda df: df["answer"].str.extract(r"#### (.*)"),
    names=SplitConfig(train="main", test="main"),
)

_HOTPOTQA_CONFIG = DatasetConfig(
    name="hotpotqa/hotpot_qa",
    alias="hotpotqa",
    revision="main",
    input="question",
    target="answer",
    names=SplitConfig(train="distractor", test="distractor"),
    splits=SplitConfig(train="train", test="validation"),
)


def get_input_mbbp(df):
    inputs = []
    for _, row in df.iterrows():
        expected_function_name = re.search(
            r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", row["code"], re.MULTILINE
        ).group(1)
        inp = row["text"] + f"\n\nExpected function name is '{expected_function_name}'."
        inputs.append(inp)
    return inputs


_MBPP_CONFIG = DatasetConfig(
    name="google-research-datasets/mbpp",
    alias="mbpp",
    revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    input=get_input_mbbp,
    target="",
    splits=SplitConfig(train="all", test="all"),
)

ALL_DATASETS = {
    "gsm8k": _GSM8K_CONFIG,
    "hotpotqa": _HOTPOTQA_CONFIG,
    "mbpp": _MBPP_CONFIG,
}
