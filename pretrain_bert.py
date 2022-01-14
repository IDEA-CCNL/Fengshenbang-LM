# coding=utf-8
import sys
import os
import time
import torch

from transformers import (
    TrainingArguments,
    HfArgumentParser,
    BertConfig,
    Trainer,
)
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertForPreTraining

from fengshen.utils.cnnl_args import CNNLTrainningArguments
from fengshen.data.megatron_dataloader import bert_dataset

if __name__ == "__main__":
    config = BertConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        layer_norm_eps=1e-05,
    )
    model = BertForPreTraining(config=config)
    print(model.num_parameters())
    tokenizer = BertTokenizer.from_pretrained(
        "/cognitive_comp/gaoxinyu/transformers/gxy_test/model")

    parser = HfArgumentParser((TrainingArguments, CNNLTrainningArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        training_args, cnnl_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        training_args, cnnl_args = parser.parse_args_into_dataclasses()

    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print('> compiling dataset index builder ...')
        from fengshen.data.megatron_dataloader.dataset_utils import compile_helper
        compile_helper()
        print('>>> done with dataset index builder. Compilation time: {:.3f} '
              'seconds'.format(time.time() - start_time), flush=True)
    bert_dataset.tokenizer = tokenizer
    train_dataset, eval_dataset = bert_dataset.build_train_valid_test_datasets(
        data_prefix=cnnl_args.megatron_data_path,
        data_impl=cnnl_args.megatron_data_impl,
        splits_string=cnnl_args.megatron_splits_string,
        train_valid_test_num_samples=None,
        max_seq_length=512,
        masked_lm_prob=0.15,
        short_seq_prob=0.1,
        seed=cnnl_args.megatron_seed,
        skip_warmup=False,
        binary_head=cnnl_args.megatron_binary_head)

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
