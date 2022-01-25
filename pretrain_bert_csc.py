import os
import sys
import time

import torch
from transformers import (
    HfArgumentParser,
    MegatronBertConfig,
    MegatronBertForMaskedLM,
    Trainer,
    TrainingArguments,
)

from fengshen.data.mmap_dataloader.dataset_utils import (
    DSET_TYPE_BERT,
    DSET_TYPE_BERT_CN,
    DSET_TYPE_BERT_CN_WWM,
    build_train_valid_test_datasets,
)
from fengshen.global_vars import get_tokenizer, set_global_variables
from fengshen.utils.ccnl_args import CCNLTrainningArguments

pretrain_model_dir = "/cognitive_comp/ganruyi/hf_models/erlangshen_1.3B"

if __name__ == "__main__":
    config = MegatronBertConfig.from_pretrained(pretrain_model_dir)
    # model = MegatronBertForMaskedLM.from_pretrained(pretrain_model_dir)
    model = MegatronBertForMaskedLM.from_pretrained(pretrain_model_dir)
    # model.resize_token_embeddings(12000)
    print(model.num_parameters())

    parser = HfArgumentParser((TrainingArguments, CCNLTrainningArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        training_args, ccnl_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        training_args, ccnl_args = parser.parse_args_into_dataclasses()

    ccnl_args.rank = torch.distributed.get_rank()
    ccnl_args.world_size = torch.distributed.get_world_size()
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print("> compiling dataset index builder ...")
        from fengshen.data.mmap_dataloader.dataset_utils import compile_helper

        compile_helper()
        print(
            ">>> done with dataset index builder. Compilation time: {:.3f} "
            "seconds".format(time.time() - start_time),
            flush=True,
        )

    set_global_variables(ccnl_args)
    tokenizer = get_tokenizer()
    dataset_type = DSET_TYPE_BERT
    if ccnl_args.tokenizer_type == "BertCNWWMTokenizer":
        dataset_type = DSET_TYPE_BERT_CN_WWM
    elif ccnl_args.tokenizer_type == "BertCNTokenizer":
        dataset_type = DSET_TYPE_BERT_CN

    # Number of train/valid/test samples.
    # if args.train_samples:
    #     train_samples = args.train_samples
    # else:
    #     train_samples = args.train_iters * args.global_batch_size
    # eval_iters = (args.train_iters // args.eval_interval + 1) * \
    #                 args.eval_iters
    # test_iters = args.eval_iters
    # train_val_test_num_samples = [train_samples,
    #                                 eval_iters * args.global_batch_size,
    #                                 test_iters * args.global_batch_size]

    train_dataset, eval_dataset, _ = build_train_valid_test_datasets(
        data_prefix=ccnl_args.megatron_data_path,
        data_impl=ccnl_args.megatron_data_impl,
        splits_string=ccnl_args.split,
        train_valid_test_num_samples=None,
        max_seq_length=512,
        masked_lm_prob=0.15,
        short_seq_prob=0.1,
        seed=ccnl_args.megatron_seed,
        skip_warmup=False,
        binary_head=ccnl_args.megatron_binary_head,
        dataset_type=dataset_type,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=None,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
