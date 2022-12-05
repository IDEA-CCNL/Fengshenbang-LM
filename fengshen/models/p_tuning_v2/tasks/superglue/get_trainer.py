import logging
import os
import random
import sys
from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from model.utils import get_model, TaskType
from tasks.superglue.dataset import SuperGlueDataset
from tasks.superglue.dataset_csqa import CSQADataset
from tasks.superglue.dataset_obqa import OBQADataset
from tasks.superglue.dataset_rsqa import RSQADataset
from tasks.superglue.dataset_siqa import SIQADataset
from tasks.superglue.dataset_piqa import PIQADataset
from tasks.superglue.dataset_medqa import MEDQADataset

from training.trainer_base import BaseTrainer
from training.trainer_exp import ExponentialTrainer
import pdb
logger = logging.getLogger(__name__)

def get_trainer(args):
    model_args, data_args, training_args, _ = args

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )
    if data_args.dataset_name == "csqa":
        dataset = CSQADataset(tokenizer, data_args, training_args)
    elif data_args.dataset_name == "obqa":
        dataset = OBQADataset(tokenizer, data_args, training_args)
    elif data_args.dataset_name == "medqa":
        dataset = MEDQADataset(tokenizer, data_args, training_args)
        #pdb.set_trace()
    elif data_args.dataset_name == "siqa":
        dataset = SIQADataset(tokenizer, data_args, training_args)
    elif data_args.dataset_name == "piqa":
        dataset = PIQADataset(tokenizer, data_args, training_args)
    elif data_args.dataset_name == "rsqa":
        dataset = RSQADataset(tokenizer, data_args, training_args)
    else:
        dataset = SuperGlueDataset(tokenizer, data_args, training_args)

    if training_args.do_train:
        for index in random.sample(range(len(dataset.train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {dataset.train_dataset[index]}.")

    if not dataset.multiple_choice:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )

    if not dataset.multiple_choice:
        model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config)
    else:
        # pdb.set_trace()
        model = get_model(model_args, TaskType.MULTIPLE_CHOICE, config, fix_bert=False)

    # Initialize our Trainer
    training_args.save_total_limit = 1,
    training_args.load_best_model_at_end=True,
    training_args.save_strategy = "no"
    # pdb.set_trace()
    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        predict_dataset=dataset.predict_dataset if training_args.do_predict else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        test_key=dataset.test_key,
    )
    trainer.state.best_model_checkpoint = os.path.join(training_args.output_dir, "checkpoint-best")
    

    return trainer, dataset.predict_dataset
