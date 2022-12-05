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
import pdb
from tqdm import tqdm
from sklearn.metrics import f1_score

task_to_keys = {
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "rte": ("premise", "hypothesis"),
    "wic": ("processed_sentence1", None),
    "wsc": ("span2_word_text", "span1_text"),
    "copa": (None, None),
    "record": (None, None),
    "multirc": ("paragraph", "question_answer"),
    "rsqa": ("None", "None"),
}
# pdb.set_trace()
logger = logging.getLogger(__name__)


class RSQADataset():
    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args) -> None:
        super().__init__()
        # raw_datasets = load_dataset("super_glue", data_args.dataset_name)
        raw_datasets = load_dataset(
                path="../tasks/rsqa_2views.py",
                cache_dir="./data_2views_rsqa/")
        #pdb.set_trace()
        self.tokenizer = tokenizer
        self.data_args = data_args

        self.multiple_choice = True#data_args.dataset_name in ["copa", "csqa"]

        # if data_args.dataset_name == "record":
        #     self.num_labels = 2
        #     self.label_list = ["0", "1"]
        # elif not self.multiple_choice:
        self.label_list = ["0", "1", "2", "3", "4"]#raw_datasets["train"].features["labels"].names
        self.num_labels = len(self.label_list)
        # else:
        #     self.num_labels = 1

        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = task_to_keys[data_args.dataset_name]

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        # if not self.multiple_choice:
        #     self.label2id = {l: i for i, l in enumerate(self.label_list)}
        #     self.id2label = {id: label for label, id in self.label2id.items()}
        #     print(f"{self.label2id}")
        #     print(f"{self.id2label}")

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        # if data_args.dataset_name == "record":
        #     raw_datasets = raw_datasets.map(
        #         self.record_preprocess_function,
        #         batched=True,
        #         load_from_cache_file=not data_args.overwrite_cache,
        #         remove_columns=raw_datasets["train"].column_names,
        #         desc="Running tokenizer on dataset",
        #     )
        # else:
        raw_datasets = raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        #pdb.set_trace()
        if training_args.do_train:
            self.train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            self.eval_dataset = raw_datasets["validation"]
            if data_args.max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

        if training_args.do_predict or data_args.dataset_name is not None or data_args.test_file is not None:
            # self.predict_dataset = raw_datasets["validation"]
            self.predict_dataset = raw_datasets["test"]
            # if data_args.max_predict_samples is not None:
            #     self.predict_dataset = self.predict_dataset.select(range(data_args.max_predict_samples))

        #self.metric = load_metric("super_glue", "coqa")
        #pdb.set_trace()
        if data_args.pad_to_max_length:
            self.data_collator = default_data_collator
        elif training_args.fp16:
            self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

        self.test_key = "accuracy" #if data_args.dataset_name not in ["record", "multirc"] else "f1"

    def preprocess_function(self, examples):
        #pdb.set_trace()
        L = len(examples["id"])
        #examples["qc_meaning"] = []
        examples["text_a"] = []
        examples["text_b"] = []
        examples["text_c"] = []
        examples["text_d"] = []
        examples["text_e"] = []
        examples["individual_view_triplets_a"] = []
        examples["individual_view_triplets_b"] = []
        examples["individual_view_triplets_c"] = []
        examples["individual_view_triplets_d"] = []
        examples["individual_view_triplets_e"] = []
        examples["latent_view_triplets_a"] = []
        examples["latent_view_triplets_b"] = []
        examples["latent_view_triplets_c"] = []
        examples["latent_view_triplets_d"] = []
        examples["latent_view_triplets_e"] = []
        examples["group_view_triplets_a"] = []
        examples["group_view_triplets_b"] = []
        examples["group_view_triplets_c"] = []
        examples["group_view_triplets_d"] = []
        examples["group_view_triplets_e"] = []
        examples["retri_view_triplets_a"] = []
        examples["retri_view_triplets_b"] = []
        examples["retri_view_triplets_c"] = []
        examples["retri_view_triplets_d"] = []
        examples["retri_view_triplets_e"] = []
        examples["selected_individual_view_triplets_a"] = []
        examples["selected_individual_view_triplets_b"] = []
        examples["selected_individual_view_triplets_c"] = []
        examples["selected_individual_view_triplets_d"] = []
        examples["selected_individual_view_triplets_e"] = []
        examples["meaning_view_triplets_a"] = []
        examples["meaning_view_triplets_b"] = []
        examples["meaning_view_triplets_c"] = []
        examples["meaning_view_triplets_d"] = []
        examples["meaning_view_triplets_e"] = []
        for kk in tqdm(range(L), total = L):
            examples["text_a"].append(examples["question_choice_A"][kk])
            examples["text_b"].append(examples["question_choice_B"][kk])
            examples["text_c"].append(examples["question_choice_C"][kk])
            examples["text_d"].append(examples["question_choice_D"][kk])
            examples["text_e"].append(examples["question_choice_E"][kk])

            #individual view triplets
            examples["individual_view_triplets_a"].append("individual view triplets: {}".format(
                examples['triplets_a'][kk]['individual_view_triplets']))
            examples["individual_view_triplets_b"].append("individual view triplets: {}".format(
                examples['triplets_b'][kk]['individual_view_triplets']))
            examples["individual_view_triplets_c"].append("individual view triplets: {}".format(
                examples['triplets_c'][kk]['individual_view_triplets']))
            examples["individual_view_triplets_d"].append("individual view triplets: {}".format(
                examples['triplets_d'][kk]['individual_view_triplets']))
            examples["individual_view_triplets_e"].append("individual view triplets: {}".format(
                examples['triplets_e'][kk]['individual_view_triplets']))   
            #selected individual view triplets
            examples["selected_individual_view_triplets_a"].append("selected individual view triplets: {}".format(
                examples['triplets_a'][kk]['selected_individual_view_triplets']))   
            examples["selected_individual_view_triplets_b"].append("selected individual view triplets: {}".format(
                examples['triplets_b'][kk]['selected_individual_view_triplets'])) 
            examples["selected_individual_view_triplets_c"].append("selected individual view triplets: {}".format(
                examples['triplets_c'][kk]['selected_individual_view_triplets'])) 
            examples["selected_individual_view_triplets_d"].append("selected individual view triplets: {}".format(
                examples['triplets_d'][kk]['selected_individual_view_triplets'])) 
            examples["selected_individual_view_triplets_e"].append("selected individual view triplets: {}".format(
                examples['triplets_e'][kk]['selected_individual_view_triplets'])) 

            #latent view triplets
            examples["latent_view_triplets_a"].append("latent view triplets: {}".format(
                examples['triplets_a'][kk]['latent_view_triplets']))
            examples["latent_view_triplets_b"].append("latent view triplets: {}".format(
                examples['triplets_b'][kk]['latent_view_triplets']))
            examples["latent_view_triplets_c"].append("latent view triplets: {}".format(
                examples['triplets_c'][kk]['latent_view_triplets']))
            examples["latent_view_triplets_d"].append("latent view triplets: {}".format(
                examples['triplets_d'][kk]['latent_view_triplets']))
            examples["latent_view_triplets_e"].append("latent view triplets: {}".format(
                examples['triplets_e'][kk]['latent_view_triplets']))
            #group view triplets
            examples["group_view_triplets_a"].append("group view triplets: {}".format(
                examples['triplets_a'][kk]['group_view_triplets']))
            examples["group_view_triplets_b"].append("group view triplets: {}".format(
                examples['triplets_b'][kk]['group_view_triplets']))
            examples["group_view_triplets_c"].append("group view triplets: {}".format(
                examples['triplets_c'][kk]['group_view_triplets']))
            examples["group_view_triplets_d"].append("group view triplets: {}".format(
                examples['triplets_d'][kk]['group_view_triplets']))
            examples["group_view_triplets_e"].append("group view triplets: {}".format(
                examples['triplets_e'][kk]['group_view_triplets']))

            #retri view triplets
            examples["retri_view_triplets_a"].append("retrieval view triplets: {}".format(
                examples['triplets_a'][kk]['retri_view_triplets']))
            examples["retri_view_triplets_b"].append("retrieval view triplets: {}".format(
                examples['triplets_b'][kk]['retri_view_triplets']))
            examples["retri_view_triplets_c"].append("retrieval view triplets: {}".format(
                examples['triplets_c'][kk]['retri_view_triplets']))
            examples["retri_view_triplets_d"].append("retrieval view triplets: {}".format(
                examples['triplets_d'][kk]['retri_view_triplets']))
            examples["retri_view_triplets_e"].append("retrieval view triplets: {}".format(
                examples['triplets_e'][kk]['retri_view_triplets']))

            #meaning view triplets
            examples["meaning_view_triplets_a"].append("meaning view triplets: {}".format(
                examples['triplets_a'][kk]['meaning_view_triplets']))
            examples["meaning_view_triplets_b"].append("meaning view triplets: {}".format(
                examples['triplets_b'][kk]['meaning_view_triplets']))
            examples["meaning_view_triplets_c"].append("meaning view triplets: {}".format(
                examples['triplets_c'][kk]['meaning_view_triplets']))
            examples["meaning_view_triplets_d"].append("meaning view triplets: {}".format(
                examples['triplets_d'][kk]['meaning_view_triplets']))
            examples["meaning_view_triplets_e"].append("meaning view triplets: {}".format(
                examples['triplets_e'][kk]['meaning_view_triplets']))

        #original input
        result1 = self.tokenizer(examples["text_a"], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
        result2 = self.tokenizer(examples["text_b"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result3 = self.tokenizer(examples["text_c"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result4 = self.tokenizer(examples["text_d"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result5 = self.tokenizer(examples["text_e"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result = {}  
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in result1 and key in result2 and key and key in result3 and key in result4 and key in result5:
                result[key] = []
                for value1, value2, value3, value4, value5 in zip(result1[key], result2[key], result3[key], result4[key], result5[key]):
                    result[key].append([value1, value2, value3, value4, value5])
        
        #individual view triplets
        result1 = self.tokenizer(examples["selected_individual_view_triplets_a"], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
        result2 = self.tokenizer(examples["selected_individual_view_triplets_b"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result3 = self.tokenizer(examples["selected_individual_view_triplets_c"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result4 = self.tokenizer(examples["selected_individual_view_triplets_d"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result5 = self.tokenizer(examples["selected_individual_view_triplets_e"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in result1 and key in result2 and key and key in result3 and key in result4 and key in result5:
                result["selected_individual_view_triplets_{}".format(key)] = []
                for value1, value2, value3, value4, value5 in zip(result1[key], result2[key], result3[key], result4[key], result5[key]):
                    result["selected_individual_view_triplets_{}".format(key)].append([value1, value2, value3, value4, value5])
        
        #individual view triplets
        result1 = self.tokenizer(examples["individual_view_triplets_a"], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
        result2 = self.tokenizer(examples["individual_view_triplets_b"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result3 = self.tokenizer(examples["individual_view_triplets_c"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result4 = self.tokenizer(examples["individual_view_triplets_d"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result5 = self.tokenizer(examples["individual_view_triplets_e"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in result1 and key in result2 and key and key in result3 and key in result4 and key in result5:
                result["individual_view_triplets_{}".format(key)] = []
                for value1, value2, value3, value4, value5 in zip(result1[key], result2[key], result3[key], result4[key], result5[key]):
                    result["individual_view_triplets_{}".format(key)].append([value1, value2, value3, value4, value5])
        #group view triplets
        result1 = self.tokenizer(examples["text_a"], examples["group_view_triplets_a"], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
        result2 = self.tokenizer(examples["text_b"], examples["group_view_triplets_b"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result3 = self.tokenizer(examples["text_c"], examples["group_view_triplets_c"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result4 = self.tokenizer(examples["text_d"], examples["group_view_triplets_d"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result5 = self.tokenizer(examples["text_e"], examples["group_view_triplets_e"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in result1 and key in result2 and key and key in result3 and key in result4 and key in result5:
                result["group_view_triplets_{}".format(key)] = []
                for value1, value2, value3, value4, value5 in zip(result1[key], result2[key], result3[key], result4[key], result5[key]):
                    result["group_view_triplets_{}".format(key)].append([value1, value2, value3, value4, value5])
        #latent view triplets
        result1 = self.tokenizer(examples["latent_view_triplets_a"], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
        result2 = self.tokenizer(examples["latent_view_triplets_b"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result3 = self.tokenizer(examples["latent_view_triplets_c"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result4 = self.tokenizer(examples["latent_view_triplets_d"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result5 = self.tokenizer(examples["latent_view_triplets_e"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in result1 and key in result2 and key and key in result3 and key in result4 and key in result5:
                result["latent_view_triplets_{}".format(key)] = []
                for value1, value2, value3, value4, value5 in zip(result1[key], result2[key], result3[key], result4[key], result5[key]):
                    result["latent_view_triplets_{}".format(key)].append([value1, value2, value3, value4, value5])
        
        #selected individual view triplets
        result1 = self.tokenizer(examples["text_a"], examples["selected_individual_view_triplets_a"], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
        result2 = self.tokenizer(examples["text_b"], examples["selected_individual_view_triplets_b"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result3 = self.tokenizer(examples["text_c"], examples["selected_individual_view_triplets_c"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result4 = self.tokenizer(examples["text_d"], examples["selected_individual_view_triplets_d"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result5 = self.tokenizer(examples["text_e"], examples["selected_individual_view_triplets_e"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in result1 and key in result2 and key and key in result3 and key in result4 and key in result5:
                result["combine_selected_individual_view_triplets_{}".format(key)] = []
                for value1, value2, value3, value4, value5 in zip(result1[key], result2[key], result3[key], result4[key], result5[key]):
                    result["combine_selected_individual_view_triplets_{}".format(key)].append([value1, value2, value3, value4, value5])

        #individual view triplets
        result1 = self.tokenizer(examples["text_a"], examples["individual_view_triplets_a"], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
        result2 = self.tokenizer(examples["text_b"], examples["individual_view_triplets_b"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result3 = self.tokenizer(examples["text_c"], examples["individual_view_triplets_c"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result4 = self.tokenizer(examples["text_d"], examples["individual_view_triplets_d"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result5 = self.tokenizer(examples["text_e"], examples["individual_view_triplets_e"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in result1 and key in result2 and key and key in result3 and key in result4 and key in result5:
                result["combine_individual_view_triplets_{}".format(key)] = []
                for value1, value2, value3, value4, value5 in zip(result1[key], result2[key], result3[key], result4[key], result5[key]):
                    result["combine_individual_view_triplets_{}".format(key)].append([value1, value2, value3, value4, value5])
        #group view triplets
        result1 = self.tokenizer(examples["text_a"], examples["group_view_triplets_a"], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
        result2 = self.tokenizer(examples["text_b"], examples["group_view_triplets_b"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result3 = self.tokenizer(examples["text_c"], examples["group_view_triplets_c"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result4 = self.tokenizer(examples["text_d"], examples["group_view_triplets_d"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result5 = self.tokenizer(examples["text_e"], examples["group_view_triplets_e"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in result1 and key in result2 and key and key in result3 and key in result4 and key in result5:
                result["combine_group_view_triplets_{}".format(key)] = []
                for value1, value2, value3, value4, value5 in zip(result1[key], result2[key], result3[key], result4[key], result5[key]):
                    result["combine_group_view_triplets_{}".format(key)].append([value1, value2, value3, value4, value5])
        
        #retri view triplets
        result1 = self.tokenizer(examples["text_a"], examples["retri_view_triplets_a"], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
        result2 = self.tokenizer(examples["text_b"], examples["retri_view_triplets_b"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result3 = self.tokenizer(examples["text_c"], examples["retri_view_triplets_c"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result4 = self.tokenizer(examples["text_d"], examples["retri_view_triplets_d"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result5 = self.tokenizer(examples["text_e"], examples["retri_view_triplets_e"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in result1 and key in result2 and key and key in result3 and key in result4 and key in result5:
                result["combine_retri_view_triplets_{}".format(key)] = []
                for value1, value2, value3, value4, value5 in zip(result1[key], result2[key], result3[key], result4[key], result5[key]):
                    result["combine_retri_view_triplets_{}".format(key)].append([value1, value2, value3, value4, value5])
        
        # #meaning view triplets
        result1 = self.tokenizer(examples["text_a"], examples["meaning_view_triplets_a"], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
        result2 = self.tokenizer(examples["text_b"], examples["meaning_view_triplets_b"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result3 = self.tokenizer(examples["text_c"], examples["meaning_view_triplets_c"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result4 = self.tokenizer(examples["text_d"], examples["meaning_view_triplets_d"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result5 = self.tokenizer(examples["text_e"], examples["meaning_view_triplets_e"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in result1 and key in result2 and key and key in result3 and key in result4 and key in result5:
                result["combine_meaning_view_triplets_{}".format(key)] = []
                for value1, value2, value3, value4, value5 in zip(result1[key], result2[key], result3[key], result4[key], result5[key]):
                    result["combine_meaning_view_triplets_{}".format(key)].append([value1, value2, value3, value4, value5])

        # #latent view triplets
        result1 = self.tokenizer(examples["text_a"], examples["latent_view_triplets_a"], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
        result2 = self.tokenizer(examples["text_b"], examples["latent_view_triplets_b"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result3 = self.tokenizer(examples["text_c"], examples["latent_view_triplets_c"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result4 = self.tokenizer(examples["text_d"], examples["latent_view_triplets_d"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result5 = self.tokenizer(examples["text_e"], examples["latent_view_triplets_e"], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in result1 and key in result2 and key and key in result3 and key in result4 and key in result5:
                result["combine_latent_view_triplets_{}".format(key)] = []
                for value1, value2, value3, value4, value5 in zip(result1[key], result2[key], result3[key], result4[key], result5[key]):
                    result["combine_latent_view_triplets_{}".format(key)].append([value1, value2, value3, value4, value5])
        # #two views
        # #individual + group
        result1 = self.tokenizer(examples["text_a"],[x+' [SEP] '+y for x,y in zip(examples["individual_view_triplets_a"],examples["group_view_triplets_a"])], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
        result2 = self.tokenizer(examples["text_b"],[x+' [SEP] '+y for x,y in zip(examples["individual_view_triplets_b"],examples["group_view_triplets_b"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result3 = self.tokenizer(examples["text_c"],[x+' [SEP] '+y for x,y in zip(examples["individual_view_triplets_c"],examples["group_view_triplets_c"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result4 = self.tokenizer(examples["text_d"],[x+' [SEP] '+y for x,y in zip(examples["individual_view_triplets_d"],examples["group_view_triplets_d"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result5 = self.tokenizer(examples["text_e"],[x+' [SEP] '+y for x,y in zip(examples["individual_view_triplets_e"],examples["group_view_triplets_e"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in result1 and key in result2 and key and key in result3 and key in result4 and key in result5:
                result["tv_individual_group_triplets_{}".format(key)] = []
                for value1, value2, value3, value4, value5 in zip(result1[key], result2[key], result3[key], result4[key], result5[key]):
                    result["tv_individual_group_triplets_{}".format(key)].append([value1, value2, value3, value4, value5])
        #individual + retri
        result1 = self.tokenizer(examples["text_a"],[x+' [SEP] '+y for x,y in zip(examples["selected_individual_view_triplets_a"],examples["retri_view_triplets_a"])], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
        result2 = self.tokenizer(examples["text_b"],[x+' [SEP] '+y for x,y in zip(examples["selected_individual_view_triplets_b"],examples["retri_view_triplets_b"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result3 = self.tokenizer(examples["text_c"],[x+' [SEP] '+y for x,y in zip(examples["selected_individual_view_triplets_c"],examples["retri_view_triplets_c"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result4 = self.tokenizer(examples["text_d"],[x+' [SEP] '+y for x,y in zip(examples["selected_individual_view_triplets_d"],examples["retri_view_triplets_d"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        result5 = self.tokenizer(examples["text_e"],[x+' [SEP] '+y for x,y in zip(examples["selected_individual_view_triplets_e"],examples["retri_view_triplets_e"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        for key in ["input_ids", "attention_mask", "token_type_ids"]:
            if key in result1 and key in result2 and key and key in result3 and key in result4 and key in result5:
                result["tv_individual_retri_triplets_{}".format(key)] = []
                for value1, value2, value3, value4, value5 in zip(result1[key], result2[key], result3[key], result4[key], result5[key]):
                    result["tv_individual_retri_triplets_{}".format(key)].append([value1, value2, value3, value4, value5])
        
        #individual + meaning
        # result1 = self.tokenizer(examples["text_a"],[x+' [SEP] '+y for x,y in zip(examples["selected_individual_view_triplets_a"],examples["meaning_view_triplets_a"])], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
        # result2 = self.tokenizer(examples["text_b"],[x+' [SEP] '+y for x,y in zip(examples["selected_individual_view_triplets_b"],examples["meaning_view_triplets_b"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # result3 = self.tokenizer(examples["text_c"],[x+' [SEP] '+y for x,y in zip(examples["selected_individual_view_triplets_c"],examples["meaning_view_triplets_c"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # result4 = self.tokenizer(examples["text_d"],[x+' [SEP] '+y for x,y in zip(examples["selected_individual_view_triplets_d"],examples["meaning_view_triplets_d"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # result5 = self.tokenizer(examples["text_e"],[x+' [SEP] '+y for x,y in zip(examples["selected_individual_view_triplets_e"],examples["meaning_view_triplets_e"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # for key in ["input_ids", "attention_mask", "token_type_ids"]:
        #     if key in result1 and key in result2 and key and key in result3 and key in result4 and key in result5:
        #         result["tv_individual_meaning_triplets_{}".format(key)] = []
        #         for value1, value2, value3, value4, value5 in zip(result1[key], result2[key], result3[key], result4[key], result5[key]):
        #             result["tv_individual_meaning_triplets_{}".format(key)].append([value1, value2, value3, value4, value5])

        # #individual + latent
        # result1 = self.tokenizer(examples["text_a"], [x+' [SEP] '+y for x,y in zip(examples["selected_individual_view_triplets_a"], examples["latent_view_triplets_a"])], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
        # result2 = self.tokenizer(examples["text_b"], [x+' [SEP] '+y for x,y in zip(examples["selected_individual_view_triplets_b"], examples["latent_view_triplets_b"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # result3 = self.tokenizer(examples["text_c"], [x+' [SEP] '+y for x,y in zip(examples["selected_individual_view_triplets_c"], examples["latent_view_triplets_c"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # result4 = self.tokenizer(examples["text_d"], [x+' [SEP] '+y for x,y in zip(examples["selected_individual_view_triplets_d"], examples["latent_view_triplets_d"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # result5 = self.tokenizer(examples["text_e"], [x+' [SEP] '+y for x,y in zip(examples["selected_individual_view_triplets_e"], examples["latent_view_triplets_e"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # for key in ["input_ids", "attention_mask", "token_type_ids"]:
        #     if key in result1 and key in result2 and key and key in result3 and key in result4 and key in result5:
        #         result["tv_individual_latent_triplets_{}".format(key)] = []
        #         for value1, value2, value3, value4, value5 in zip(result1[key], result2[key], result3[key], result4[key], result5[key]):
        #             result["tv_individual_latent_triplets_{}".format(key)].append([value1, value2, value3, value4, value5])
        # #group + latent
        # result1 = self.tokenizer(examples["text_a"], [x+' [SEP] '+y for x,y in zip(examples["group_view_triplets_a"],examples["latent_view_triplets_a"])], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
        # result2 = self.tokenizer(examples["text_b"], [x+' [SEP] '+y for x,y in zip(examples["group_view_triplets_b"],examples["latent_view_triplets_b"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # result3 = self.tokenizer(examples["text_c"], [x+' [SEP] '+y for x,y in zip(examples["group_view_triplets_c"],examples["latent_view_triplets_c"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # result4 = self.tokenizer(examples["text_d"], [x+' [SEP] '+y for x,y in zip(examples["group_view_triplets_d"],examples["latent_view_triplets_d"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # result5 = self.tokenizer(examples["text_e"], [x+' [SEP] '+y for x,y in zip(examples["group_view_triplets_e"],examples["latent_view_triplets_e"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # for key in ["input_ids", "attention_mask", "token_type_ids"]:
        #     if key in result1 and key in result2 and key and key in result3 and key in result4 and key in result5:
        #         result["tv_group_latent_triplets_{}".format(key)] = []
        #         for value1, value2, value3, value4, value5 in zip(result1[key], result2[key], result3[key], result4[key], result5[key]):
        #             result["tv_group_latent_triplets_{}".format(key)].append([value1, value2, value3, value4, value5])
        # #individual + group + latent
        # result1 = self.tokenizer(examples["text_a"], [x+' [SEP] '+y+' [SEP] '+z for x,y,z in zip(examples["selected_individual_view_triplets_a"],examples["group_view_triplets_a"],examples["latent_view_triplets_a"])], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
        # result2 = self.tokenizer(examples["text_b"], [x+' [SEP] '+y+' [SEP] '+z for x,y,z in zip(examples["selected_individual_view_triplets_b"],examples["group_view_triplets_b"],examples["latent_view_triplets_b"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # result3 = self.tokenizer(examples["text_c"], [x+' [SEP] '+y+' [SEP] '+z for x,y,z in zip(examples["selected_individual_view_triplets_c"],examples["group_view_triplets_c"],examples["latent_view_triplets_c"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # result4 = self.tokenizer(examples["text_d"], [x+' [SEP] '+y+' [SEP] '+z for x,y,z in zip(examples["selected_individual_view_triplets_d"],examples["group_view_triplets_d"],examples["latent_view_triplets_d"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # result5 = self.tokenizer(examples["text_e"], [x+' [SEP] '+y+' [SEP] '+z for x,y,z in zip(examples["selected_individual_view_triplets_e"],examples["group_view_triplets_e"],examples["latent_view_triplets_e"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # for key in ["input_ids", "attention_mask", "token_type_ids"]:
        #     if key in result1 and key in result2 and key and key in result3 and key in result4 and key in result5:
        #         result["v3_individual_group_triplets_{}".format(key)] = []
        #         for value1, value2, value3, value4, value5 in zip(result1[key], result2[key], result3[key], result4[key], result5[key]):
        #             result["v3_individual_group_triplets_{}".format(key)].append([value1, value2, value3, value4, value5])

        # #individual + retri + meaning
        # result1 = self.tokenizer(examples["text_a"], [x+' [SEP] '+y+' [SEP] '+z for x,y,z in zip(examples["selected_individual_view_triplets_a"],examples["retri_view_triplets_a"],examples["meaning_view_triplets_a"])], padding=self.padding, max_length=self.max_seq_length, truncation=True) 
        # result2 = self.tokenizer(examples["text_b"], [x+' [SEP] '+y+' [SEP] '+z for x,y,z in zip(examples["selected_individual_view_triplets_b"],examples["retri_view_triplets_b"],examples["meaning_view_triplets_b"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # result3 = self.tokenizer(examples["text_c"], [x+' [SEP] '+y+' [SEP] '+z for x,y,z in zip(examples["selected_individual_view_triplets_c"],examples["retri_view_triplets_c"],examples["meaning_view_triplets_c"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # result4 = self.tokenizer(examples["text_d"], [x+' [SEP] '+y+' [SEP] '+z for x,y,z in zip(examples["selected_individual_view_triplets_d"],examples["retri_view_triplets_d"],examples["meaning_view_triplets_d"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # result5 = self.tokenizer(examples["text_e"], [x+' [SEP] '+y+' [SEP] '+z for x,y,z in zip(examples["selected_individual_view_triplets_e"],examples["retri_view_triplets_e"],examples["meaning_view_triplets_e"])], padding=self.padding, max_length=self.max_seq_length, truncation=True)
        # for key in ["input_ids", "attention_mask", "token_type_ids"]:
        #     if key in result1 and key in result2 and key and key in result3 and key in result4 and key in result5:
        #         result["v3_individual_retri_meaning_triplets_{}".format(key)] = []
        #         for value1, value2, value3, value4, value5 in zip(result1[key], result2[key], result3[key], result4[key], result5[key]):
        #             result["v3_individual_retri_meaning_triplets_{}".format(key)].append([value1, value2, value3, value4, value5])
        return result

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)

        # if self.data_args.dataset_name == "record":
        #     return self.reocrd_compute_metrics(p)

        # if self.data_args.dataset_name == "multirc":
        #     from sklearn.metrics import f1_score
        #     return {"f1": f1_score(preds, p.label_ids)}

        # if self.data_args.dataset_name is not None:
        #     result = self.metric.compute(predictions=preds, references=p.label_ids)
        #     if len(result) > 1:
        #         result["combined_score"] = np.mean(list(result.values())).item()
        #     return result
        # elif self.is_regression:
        #     return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        # else:
        #     return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item(), "f1": f1_score(preds, p.label_ids)}
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item(), "f1": f1_score(preds, p.label_ids,average='macro')}

    # def reocrd_compute_metrics(self, p: EvalPrediction):
    #     from tasks.superglue.utils import f1_score, exact_match_score, metric_max_over_ground_truths
    #     probs = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    #     examples = self.eval_dataset
    #     qid2pred = defaultdict(list)
    #     qid2ans = {}
    #     for prob, example in zip(probs, examples):
    #         qid = example['question_id']
    #         qid2pred[qid].append((prob[1], example['entity']))
    #         if qid not in qid2ans:
    #             qid2ans[qid] = example['answers']
    #     n_correct, n_total = 0, 0
    #     f1, em = 0, 0
    #     for qid in qid2pred:
    #         preds = sorted(qid2pred[qid], reverse=True)
    #         entity = preds[0][1]
    #         n_total += 1
    #         n_correct += (entity in qid2ans[qid])
    #         f1 += metric_max_over_ground_truths(f1_score, entity, qid2ans[qid])
    #         em += metric_max_over_ground_truths(exact_match_score, entity, qid2ans[qid])
    #     acc = n_correct / n_total
    #     f1 = f1 / n_total
    #     em = em / n_total
    #     return {'f1': f1, 'exact_match': em}

    # def record_preprocess_function(self, examples, split="train"):
    #     results = {
    #         "index": list(),
    #         "question_id": list(),
    #         "input_ids": list(),
    #         "attention_mask": list(),
    #         "token_type_ids": list(),
    #         "label": list(),
    #         "entity": list(),
    #         "answers": list()
    #     }
    #     for idx, passage in enumerate(examples["passage"]):
    #         query, entities, answers =  examples["query"][idx], examples["entities"][idx], examples["answers"][idx]
    #         index = examples["idx"][idx]
    #         passage = passage.replace("@highlight\n", "- ")
            
    #         for ent_idx, ent in enumerate(entities):
    #             question = query.replace("@placeholder", ent)
    #             result = self.tokenizer(passage, question, padding=self.padding, max_length=self.max_seq_length, truncation=True)
    #             label = 1 if ent in answers else 0

    #             results["input_ids"].append(result["input_ids"])
    #             results["attention_mask"].append(result["attention_mask"])
    #             if "token_type_ids" in result: results["token_type_ids"].append(result["token_type_ids"])
    #             results["label"].append(label)
    #             results["index"].append(index)
    #             results["question_id"].append(index["query"])
    #             results["entity"].append(ent)
    #             results["answers"].append(answers)

    #     return results
