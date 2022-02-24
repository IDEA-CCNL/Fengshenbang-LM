# coding=utf-8
import sys
import os

# 临时这样干
sys.path.append(r"../")

from transformers import (
    TrainingArguments,
    HfArgumentParser,
    BertConfig,
)
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertForPreTraining

from fengshen.trainer.cnnl_trainer import CNNLTrainer
from fengshen.trainer.cnnl_args import CNNLTrainningArguments

if __name__ == "__main__":
    config = BertConfig.from_pretrained("bert-base-uncased")
    model = BertForPreTraining(config=config)
    print(model.num_parameters())
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    training_args = TrainingArguments(
        output_dir="./bert_base",
        overwrite_output_dir=True,
        num_train_epochs=20,
        per_gpu_train_batch_size=8,
        save_steps=2000,
        save_total_limit=2,
    )

    parser = HfArgumentParser((TrainingArguments, CNNLTrainningArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        training_args, cnnl_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        training_args, cnnl_args = parser.parse_args_into_dataclasses()

    trainer = CNNLTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        cnnl_args=cnnl_args,
    )
    trainer.train()
