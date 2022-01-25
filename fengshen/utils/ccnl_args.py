from dataclasses import dataclass, field
from typing import List


@dataclass
class CCNLTrainningArguments:
    megatron_data_path: List[str] = field(
        metadata={"help": "megatron preprocess datapath, end with .idx and .bin"}
    )
    megatron_data_impl: str = field(default="mmap", metadata={"help": "impl type"})
    split: str = field(default="949,50,1")
    megatron_binary_head: bool = field(default=True)
    megatron_seed: int = field(default=1234)
    tokenizer_type: str = field(default="BertCNWWMTokenizer")
    vocab_file: str = field(default=None)
    vocab_extra_ids: int = field(default=0)
