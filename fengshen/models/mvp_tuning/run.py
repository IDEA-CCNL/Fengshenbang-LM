import logging
import os
import sys
import numpy as np
from typing import Dict

import datasets
import transformers
from transformers import set_seed, Trainer
from transformers.trainer_utils import get_last_checkpoint

from arguments import get_args
import json
from tasks.utils import *
import pdb
import torch
os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

def train(trainer, resume_from_checkpoint=None, last_checkpoint=None):
    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    # trainer.save_model()

    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.save_model()

    trainer.log_best_metrics()


def evaluate(trainer):
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

def predict(trainer, predict_dataset=None):
    if predict_dataset is None:
        logger.info("No dataset is available for testing")

    elif isinstance(predict_dataset, dict):
        
        for dataset_name, d in predict_dataset.items():
            logger.info("*** Predict: %s ***" % dataset_name)
            predictions, labels, metrics = trainer.predict(d, metric_key_prefix="predict")
            predictions = np.argmax(predictions, axis=2)

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

    else:
        
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        # pdb.set_trace()
        predictions = np.argmax(predictions, axis=1)
        preds_path = os.path.join(training_args.output_dir, 'best_test_preds.csv')
        test_file="./data/{}/statement/test.statement.jsonl".format(data_args.dataset_name)
        print(test_file,preds_path)
        with open(test_file, 'r') as f:
            test_lines = f.readlines()#json.load(f)
            # pdb.set_trace()
            test_ids = [json.loads(l)['id'] for l in test_lines]
        # pdb.set_trace()
        with open(preds_path, 'w') as f_preds:
            with torch.no_grad():
                for qid, pred, label in zip(test_ids, predictions, labels):
                    # pdb.set_trace()
                    print ('{},{},{}'.format(qid, chr(ord('A') + pred.item()), chr(ord('A') + label.item())), file=f_preds)
                    f_preds.flush()
        print("test acc: ",metrics)
        trainer.log_metrics("predict", metrics)
        # trainer.save_metrics("predict", metrics)


if __name__ == '__main__':

    args = get_args()

    _, data_args, training_args, _ = args

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    

    if not os.path.isdir("checkpoints") or not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    if data_args.task_name.lower() == "superglue":
        assert data_args.dataset_name.lower() in SUPERGLUE_DATASETS
        from tasks.superglue.get_trainer import get_trainer

    elif data_args.task_name.lower() == "glue":
        assert data_args.dataset_name.lower() in GLUE_DATASETS
        from tasks.glue.get_trainer import get_trainer

    elif data_args.task_name.lower() == "ner":
        assert data_args.dataset_name.lower() in NER_DATASETS
        from tasks.ner.get_trainer import get_trainer

    elif data_args.task_name.lower() == "srl":
        assert data_args.dataset_name.lower() in SRL_DATASETS
        from tasks.srl.get_trainer import get_trainer
    
    elif data_args.task_name.lower() == "qa":
        assert data_args.dataset_name.lower() in QA_DATASETS
        from tasks.qa.get_trainer import get_trainer
        
    else:
        raise NotImplementedError('Task {} is not implemented. Please choose a task from: {}'.format(data_args.task_name, ", ".join(TASKS)))

    set_seed(training_args.seed)

    trainer, predict_dataset = get_trainer(args)
    #pdb.set_trace()
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # if training_args.do_predict:
    #     predict(trainer, predict_dataset)

    if training_args.do_train:
        #pdb.set_trace()
        print("****** checkpoints: {} ********".format(training_args.resume_from_checkpoint))
        train(trainer, training_args.resume_from_checkpoint, last_checkpoint)
    
    if training_args.do_eval:
        evaluate(trainer)

    if training_args.do_predict:
        predict(trainer, predict_dataset)

   