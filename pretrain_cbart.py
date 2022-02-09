import os
import sys
import torch

from transformers import (
    TrainingArguments,
    HfArgumentParser,
    Trainer,
)
from fengshen.models.bart import BartForTextInfill
from transformers.models.bert.tokenization_bert import BertTokenizer
from fengshen.data.cbart_dataloader.cbart_dataset import get_train_dev_dataset, CBartDataCollator
from dataclasses import dataclass, field


@dataclass
class CBartArguments:
    model_path: str = field()
    num_labels: int = field(default=3)
    random_init_std: int = field(default=0)
    w: float = field(default=1.0)
    masked_lm: float = field(default=0)
    full_mask: float = field(default=0)
    encoder_loss_type: int = field(default=0)
    dataset: str = field(default="yelp_review")
    insert_mode: int = field(default=0)


if __name__ == '__main__':

    # 解析huggface的参数
    parser = HfArgumentParser((TrainingArguments, CBartArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        training_args, cnnl_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        training_args, args = parser.parse_args_into_dataclasses()

    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    trainset, testset = get_train_dev_dataset(args, tokenizer)
    model = BartForTextInfill.from_pretrained(args.model_path, num_labels=args.num_labels,
                                              encoder_loss_type=args.encoder_loss_type,
                                              loss_weight=args.w,)

    data_collator = CBartDataCollator(args)
    data_collator.tokenizer = tokenizer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=trainset,
        eval_dataset=testset,
        data_collator=data_collator,
    )
    model.label_weights = torch.tensor(
        trainset.label_weights, dtype=torch.half).to(training_args.device)
    trainer.train()
