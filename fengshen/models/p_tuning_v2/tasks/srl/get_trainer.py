import logging
import os
import random
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from tasks.srl.dataset import SRLDataset
from training.trainer_exp import ExponentialTrainer
from model.utils import get_model, TaskType
from tasks.utils import ADD_PREFIX_SPACE, USE_FAST

logger = logging.getLogger(__name__)

def get_trainer(args):
    model_args, data_args, training_args, _ = args

    model_type = AutoConfig.from_pretrained(model_args.model_name_or_path).model_type

    add_prefix_space = ADD_PREFIX_SPACE[model_type]

    use_fast = USE_FAST[model_type]
        
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=use_fast,
        revision=model_args.model_revision,
        add_prefix_space=add_prefix_space,
    )

    dataset = SRLDataset(tokenizer, data_args, training_args)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=dataset.num_labels,
        revision=model_args.model_revision,
    )

    if training_args.do_train:
        for index in random.sample(range(len(dataset.train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {dataset.train_dataset[index]}.")

    model = get_model(model_args, TaskType.TOKEN_CLASSIFICATION, config, fix_bert=False)
    

    trainer = ExponentialTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        predict_dataset=dataset.predict_dataset if training_args.do_predict else None,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        compute_metrics=dataset.compute_metrics,
        test_key="f1"
    )

    return trainer, dataset.predict_dataset