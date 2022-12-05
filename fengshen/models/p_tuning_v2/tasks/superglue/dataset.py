from datasets.load import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import logging
from collections import defaultdict

task_to_keys = {
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "rte": ("premise", "hypothesis"),
    "wic": ("processed_sentence1", None),
    "wsc": ("span2_word_text", "span1_text"),
    "copa": (None, None),
    "record": (None, None),
    "multirc": ("paragraph", "question_answer")
}

logger = logging.getLogger(__name__)


class SuperGlueDataset():
    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args) -> None:
        super().__init__()
        raw_datasets = load_dataset("super_glue", data_args.dataset_name)
        self.tokenizer = tokenizer
        self.data_args = data_args

        self.multiple_choice = data_args.dataset_name in ["copa"]

        if data_args.dataset_name == "record":
            self.num_labels = 2
            self.label_list = ["0", "1"]
        elif not self.multiple_choice:
            self.label_list = raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
        else:
            self.num_labels = 1

        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = task_to_keys[data_args.dataset_name]

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        if not self.multiple_choice:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}
            print(f"{self.label2id}")
            print(f"{self.id2label}")

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        if data_args.dataset_name == "record":
            raw_datasets = raw_datasets.map(
                self.record_preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )
        else:
            raw_datasets = raw_datasets.map(
                self.preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        if training_args.do_train:
            self.train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            self.eval_dataset = raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            self.predict_dataset = raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))

        self.metric = load_metric("super_glue", data_args.dataset_name)

        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

        self.test_key = "accuracy" if data_args.dataset_name not in ["record", "multirc"] else "f1"

    def preprocess_function(self, examples):
        # WSC
        if self.data_args.dataset_name == "wsc":
            examples["span2_word_text"] = []
            for text, span2_index, span2_word in zip(examples["text"], examples["span2_index"], examples["span2_text"]):
                if self.data_args.template_id == 0:
                    examples["span2_word_text"].append(span2_word + ": " + text)
                elif self.data_args.template_id == 1:
                    words_a = text.split()
                    words_a[span2_index] = "*" + words_a[span2_index] + "*"
                    examples["span2_word_text"].append(' '.join(words_a))

        # WiC
        if self.data_args.dataset_name == "wic":
            examples["processed_sentence1"] = []
            if self.data_args.template_id == 1:
                self.sentence2_key = "processed_sentence2"
                examples["processed_sentence2"] = []
            for sentence1, sentence2, word, start1, end1, start2, end2 in zip(examples["sentence1"], examples["sentence2"], examples["word"], examples["start1"], examples["end1"], examples["start2"], examples["end2"]):
                if self.data_args.template_id == 0: #ROBERTA
                    examples["processed_sentence1"].append(f"{sentence1} {sentence2} Does {word} have the same meaning in both sentences?")
                elif self.data_args.template_id == 1: #BERT
                    examples["processed_sentence1"].append(word + ": " + sentence1)
                    examples["processed_sentence2"].append(word + ": " + sentence2)

        # MultiRC
        if self.data_args.dataset_name == "multirc":
            examples["question_answer"] = []
            for question, asnwer in zip(examples["question"], examples["answer"]):
                examples["question_answer"].append(f"{question} {asnwer}")

        # COPA
        if self.data_args.dataset_name == "copa":
            examples["text_a"] = []
            for premise, question in zip(examples["premise"], examples["question"]):
                joiner = "because" if question == "cause" else "so"
                text_a = f"{premise} {joiner}"                    
                examples["text_a"].append(text_a)

            result1 = self.tokenizer(examples["text_a"], examples["choice1"], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
            result2 = self.tokenizer(examples["text_a"], examples["choice2"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
            result = {}  
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                if key in result1 and key in result2:
                    result[key] = []
                    for value1, value2 in zip(result1[key], result2[key]):
                        result[key].append([value1, value2])
            return result

        args = (
            (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
        )
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)

        return result

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)

        if self.data_args.dataset_name == "record":
            return self.reocrd_compute_metrics(p)

        if self.data_args.dataset_name == "multirc":
            from sklearn.metrics import f1_score
            return {"f1": f1_score(preds, p.label_ids)}

        if self.data_args.dataset_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    def reocrd_compute_metrics(self, p: EvalPrediction):
        from tasks.superglue.utils import f1_score, exact_match_score, metric_max_over_ground_truths
        probs = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        examples = self.eval_dataset
        qid2pred = defaultdict(list)
        qid2ans = {}
        for prob, example in zip(probs, examples):
            qid = example['question_id']
            qid2pred[qid].append((prob[1], example['entity']))
            if qid not in qid2ans:
                qid2ans[qid] = example['answers']
        n_correct, n_total = 0, 0
        f1, em = 0, 0
        for qid in qid2pred:
            preds = sorted(qid2pred[qid], reverse=True)
            entity = preds[0][1]
            n_total += 1
            n_correct += (entity in qid2ans[qid])
            f1 += metric_max_over_ground_truths(f1_score, entity, qid2ans[qid])
            em += metric_max_over_ground_truths(exact_match_score, entity, qid2ans[qid])
        acc = n_correct / n_total
        f1 = f1 / n_total
        em = em / n_total
        return {'f1': f1, 'exact_match': em}

    def record_preprocess_function(self, examples, split="train"):
        results = {
            "index": list(),
            "question_id": list(),
            "input_ids": list(),
            "attention_mask": list(),
            "token_type_ids": list(),
            "label": list(),
            "entity": list(),
            "answers": list()
        }
        for idx, passage in enumerate(examples["passage"]):
            query, entities, answers =  examples["query"][idx], examples["entities"][idx], examples["answers"][idx]
            index = examples["idx"][idx]
            passage = passage.replace("@highlight\n", "- ")
            
            for ent_idx, ent in enumerate(entities):
                question = query.replace("@placeholder", ent)
                result = self.tokenizer(passage, question, padding=self.padding, max_length=self.max_seq_length, truncation=True)
                label = 1 if ent in answers else 0

                results["input_ids"].append(result["input_ids"])
                results["attention_mask"].append(result["attention_mask"])
                if "token_type_ids" in result: results["token_type_ids"].append(result["token_type_ids"])
                results["label"].append(label)
                results["index"].append(index)
                results["question_id"].append(index["query"])
                results["entity"].append(ent)
                results["answers"].append(answers)

        return results
