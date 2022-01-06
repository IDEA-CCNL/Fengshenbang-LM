from dataclasses import dataclass, field
from typing import List


@dataclass
class CNNLTrainningArguments:
    megatron_data_path: str = field(
        metadata={"help": "megatron preprocess datapath, end with .idx and .bin"}
    )
    megatron_data_impl: str = field(
        default="mmap",
        metadata={"help": "impl type"}
    )
    megatron_splits_string: str = field(
        default="949,50,1"
    )
    megatron_binary_head: bool = field(
        default=True
    )
    megatron_seed: int = field(
        default=1234
    )
