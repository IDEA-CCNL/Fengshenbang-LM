import logging
import os
import random
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from tasks.qa.dataset import SQuAD
from training.trainer_qa import QuestionAnsweringTrainer
from model.utils import get_model, TaskType

logger = logging.getLogger(__name__)

def get_trainer(args):
    model_args, data_args, training_args, qa_args = args

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=2,
        revision=model_args.model_revision,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        use_fast=True,
    )

    model = get_model(model_args, TaskType.QUESTION_ANSWERING, config, fix_bert=True)
    
    dataset = SQuAD(tokenizer, data_args, training_args, qa_args)

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        eval_examples=dataset.eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        post_process_function=dataset.post_processing_function,
        compute_metrics=dataset.compute_metrics,
    )

    return trainer, dataset.predict_dataset


