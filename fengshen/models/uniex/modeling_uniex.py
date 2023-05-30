# coding=utf-8
# Copyright 2021 The IDEA Authors. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tarfile
import torch
from torch import nn
import json
from tqdm import tqdm
import os
import numpy as np
from transformers import AutoTokenizer,AutoConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import trainer, loggers
from torch.utils.data import Dataset, DataLoader
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import BertModel
from transformers import BertPreTrainedModel
import unicodedata
import re
import argparse
import copy
from typing import List, Optional
from torch import Tensor
import time
import gc
# os.environ["CUDA_VISIBLE_DEVICES"] = '7'

# pl.seed_everything(42)

def get_entity_f1(test_data,pred_data):
    corr=0
    y_true=0
    y_pred=0
    
    for i in range(len(test_data)):
        tmp_corr=0
        y_true_list=[]
        for e in test_data[i]['entity_list']:
            if (e['entity_type'],e['entity_index']) not in y_true_list:
                y_true_list.append((e['entity_type'],e['entity_index']))
                
        if y_true_list==[] and 'spo_list' in test_data[i].keys():
             for spo in test_data[i]['spo_list']:
                if (spo['subject']['entity_type'],spo['subject']['entity_index']) not in y_true_list:
                    y_true_list.append((spo['subject']['entity_type'],spo['subject']['entity_index']))
                
                if (spo['object']['entity_type'],spo['object']['entity_index']) not in y_true_list:
                    y_true_list.append((spo['object']['entity_type'],spo['object']['entity_index']))

        # y_true_list=list(set(y_true_list))
        y_true+=len(y_true_list)
        
        y_pred_list=[]
        for e in pred_data[i]['entity_list']:
            if (e['entity_type'],e['entity_index']) not in y_pred_list:
                y_pred_list.append((e['entity_type'],e['entity_index']))
        
        if y_pred_list==[] and 'spo_list' in pred_data[i].keys():
             for spo in pred_data[i]['spo_list']:
                if (spo['subject']['entity_type'],spo['subject']['entity_index']) not in y_pred_list:
                    y_pred_list.append((spo['subject']['entity_type'],spo['subject']['entity_index']))
                
                if (spo['object']['entity_type'],spo['object']['entity_index']) not in y_pred_list:
                    y_pred_list.append((spo['object']['entity_type'],spo['object']['entity_index']))
        
        y_pred+=len(y_pred_list)

        for e in y_pred_list:
            if e in y_true_list:
                corr+=1
    if y_pred<=0:
        precise=0
    else:
        precise = corr/y_pred
    
    if y_true<=0:
        recall=0
    else:
        recall = corr/y_true
    if precise+recall<=0:
        f1=0
    else:
        f1=2*precise*recall/(precise+recall)
    
    return f1,recall,precise


def get_entity_f1_strict(test_data,pred_data,  strict_f1=True):
    corr=0
    y_true=0
    y_pred=0
    for i in range(len(test_data)):
        tmp_corr=0
        y_true_list=[]
        y_pred_list=[]
        
        if strict_f1:
            for spo in test_data[i]['spo_list']:
                tmp=(spo['subject']['entity_type'],spo['subject']['entity_text'],spo['predicate'],spo['object']['entity_type'],spo['object']['entity_text'])
                if tmp not in y_true_list:
                    y_true_list.append(tmp)

            for spo in pred_data[i]['spo_list']:
                tmp=(spo['subject']['entity_type'],spo['subject']['entity_text'],spo['predicate'],spo['object']['entity_type'],spo['object']['entity_text'])
                if tmp not in y_pred_list:
                    y_pred_list.append(tmp)

        else:
            for spo in test_data[i]['spo_list']:
                tmp=(spo['subject']['entity_text'],spo['predicate'],spo['object']['entity_text'])
                if tmp not in y_true_list:
                    y_true_list.append(tmp)

            
            for spo in pred_data[i]['spo_list']:
                tmp=(spo['subject']['entity_text'],spo['predicate'],spo['object']['entity_text'])
                if tmp not in y_pred_list:
                    y_pred_list.append(tmp)

        y_true+=len(y_true_list)
        y_pred+=len(y_pred_list)

        for e in y_pred_list:
            if e in y_true_list:
                corr+=1
    if y_pred<=0:
        precise=0
    else:
        precise = corr/y_pred
    
    if y_true<=0:
        recall=0
    else:
        recall = corr/y_true
    if precise+recall<=0:
        f1=0
    else:
        f1=2*precise*recall/(precise+recall)
    
    return f1,recall,precise


def get_rel_f1(test_data,pred_data):
    entity_f1,_,_=get_entity_f1(test_data,pred_data)
    rel_f1,_,_=get_entity_f1_strict(test_data,pred_data,strict_f1=False)
    rel_f1_strict,rel_p,rel_r=get_entity_f1_strict(test_data,pred_data,strict_f1=True)
    return rel_f1_strict,rel_p,rel_r
    

def get_event_f1(test_data,pred_data,  strict_f1=True):
    corr_trigger=0
    y_true_trigger=0
    y_pred_trigger=0
    
    corr_args=0
    y_true_args=0
    y_pred_args=0
    for i in range(len(test_data)):
        tmp_corr=0
        y_true_list_trigger=[]
        y_pred_list_trigger=[]
        
        y_true_list_args=[]
        y_pred_list_args=[]
        

        for event in test_data[i]['event_list']:
            for a in event['args']:
                if a['entity_type']=='触发词':
                    if (a['entity_index'],a['entity_type'],event['event_type']) not in y_true_list_trigger:
                        y_true_list_trigger.append((a['entity_index'],a['entity_type'],event['event_type']))
                else:
                    if (a['entity_index'],a['entity_type'],event['event_type']) not in y_true_list_args:
                        y_true_list_args.append((a['entity_index'],a['entity_type'],event['event_type']))


        for event in pred_data[i]['event_list']:
            for a in event['args']:
                if a['entity_type']=='触发词':
                    if (a['entity_index'],a['entity_type'],event['event_type']) not in y_pred_list_trigger:
                        y_pred_list_trigger.append((a['entity_index'],a['entity_type'],event['event_type']))
                else:
                    if (a['entity_index'],a['entity_type'],event['event_type']) not in y_pred_list_args:
                        y_pred_list_args.append((a['entity_index'],a['entity_type'],event['event_type']))

        y_true_trigger+=len(y_true_list_trigger)
        y_pred_trigger+=len(y_pred_list_trigger)

        for e in y_pred_list_trigger:
            if e in y_true_list_trigger:
                corr_trigger+=1
        
        y_true_args+=len(y_true_list_args)
        y_pred_args+=len(y_pred_list_args)

        for e in y_pred_list_args:
            if e in y_true_list_args:
                corr_args+=1
                
    if y_pred_trigger<=0:
        precise_trigger=0
    else:
        precise_trigger = corr_trigger/y_pred_trigger
    
    if y_true_trigger<=0:
        recall_trigger=0
    else:
        recall_trigger = corr_trigger/y_true_trigger
    if precise_trigger+recall_trigger<=0:
        f1_trigger=0
    else:
        f1_trigger=2*precise_trigger*recall_trigger/(precise_trigger+recall_trigger)
        
    if y_pred_args<=0:
        precise_args=0
    else:
        precise_args = corr_args/y_pred_args
    
    if y_true_args<=0:
        recall_args=0
    else:
        recall_args = corr_args/y_true_args
    if precise_args+recall_args<=0:
        f1_args=0
    else:
        f1_args=2*precise_args*recall_args/(precise_args+recall_args)
        
    return f1_trigger+f1_args, f1_trigger, f1_args



class UniEXDataEncode:
    def __init__(self, tokenizer, args, used_mask=False):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.used_mask = used_mask
        self.args = args

    def search_index(self, entity_idx, text):
        start_idx_text = text[:entity_idx[0]]
        start_idx_text_encode = self.tokenizer.encode(
            start_idx_text, add_special_tokens=False)
        start_idx = len(start_idx_text_encode)

        end_idx_text = text[:entity_idx[1]+1]
        end_idx_text_encode = self.tokenizer.encode(
            end_idx_text, add_special_tokens=False)
        end_idx = len(end_idx_text_encode)-1
        return start_idx, end_idx

    def search(self, pattern, sequence):
        n = len(pattern)
        res = []
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                res.append(i)
        return res

    def get_token_type(self, sep_idx, max_length):
        token_type_ids = np.zeros(shape=(max_length,))
        for i in range(len(sep_idx)-1):
            if i % 2 == 0:
                ty = np.ones(shape=(sep_idx[i+1]-sep_idx[i],))
            else:
                ty = np.zeros(shape=(sep_idx[i+1]-sep_idx[i],))
            token_type_ids[sep_idx[i]:sep_idx[i+1]] = ty

        return token_type_ids

    def get_position_ids(self, max_length, entity_labels_idx, relation_labels_idx):
        query_length = entity_labels_idx[0]
        query_position_ids = np.arange(query_length)
        entity_position_ids = np.arange(query_length, entity_labels_idx[-1])
        for i in range(len(entity_labels_idx)-1):
            entity_position_ids[entity_labels_idx[i]-query_length:entity_labels_idx[i+1]-query_length] = np.arange(
                query_length, query_length+entity_labels_idx[i+1]-entity_labels_idx[i])
        
        if relation_labels_idx==[]:
            cur_pid=max(entity_position_ids)+1
            text_position_ids = np.arange(
                cur_pid, max_length+cur_pid-entity_labels_idx[-1])
            position_ids = list(query_position_ids) + \
                list(entity_position_ids)+list(text_position_ids)
        else:
            sep_pid = [max(entity_position_ids)+1]
            cur_pid = max(entity_position_ids)+2
            relation_position_ids = np.arange(relation_labels_idx[0], relation_labels_idx[-1])
            for i in range(len(relation_labels_idx)-1):
                relation_position_ids[relation_labels_idx[i]-relation_labels_idx[0]:relation_labels_idx[i+1]-relation_labels_idx[0]] = np.arange(
                    cur_pid, cur_pid + relation_labels_idx[i+1]-relation_labels_idx[i])
            cur_pid=max(relation_position_ids)+1
            text_position_ids = np.arange(
                cur_pid, max_length+cur_pid-relation_labels_idx[-1])
            position_ids = list(query_position_ids) +  \
                list(entity_position_ids)+sep_pid +list(relation_position_ids)+list(text_position_ids)
                
        if max_length <= 512:
            return position_ids[:max_length]
        else:
            for i in range(512, max_length):
                if position_ids[i] > 511:
                    position_ids[i] = 511
            return position_ids[:max_length]
  
        
    def get_att_mask(self, attention_mask, entity_labels_idx, relation_labels_idx, entity_type_list=None, relation_type_list=None, headtail2relation=None):
        max_length = len(attention_mask)
        attention_mask = np.array(attention_mask)
        attention_mask = np.tile(attention_mask[None, :], (max_length, 1))

        zeros = np.zeros(
            shape=(entity_labels_idx[-1]-entity_labels_idx[0], entity_labels_idx[-1]-entity_labels_idx[0]))
        attention_mask[entity_labels_idx[0]:entity_labels_idx[-1],
                        entity_labels_idx[0]:entity_labels_idx[-1]] = zeros

        attention_mask[0,1:entity_labels_idx[-1]] = np.zeros(shape=(entity_labels_idx[-1]-1,))  # 不让 【CLS】 关注到 option
        attention_mask[1:entity_labels_idx[-1],0] = np.zeros(shape=(entity_labels_idx[-1]-1,))

        for i in range(len(entity_labels_idx)-1):
            label_token_length = entity_labels_idx[i+1]-entity_labels_idx[i]
            ones = np.ones(shape=(label_token_length, label_token_length))
            attention_mask[entity_labels_idx[i]:entity_labels_idx[i+1],
                            entity_labels_idx[i]:entity_labels_idx[i+1]] = ones
        
        if relation_labels_idx == []:
            return attention_mask
        else:
            zeros = np.zeros(
                shape=(relation_labels_idx[-1]-relation_labels_idx[0], relation_labels_idx[-1]-relation_labels_idx[0]))
            attention_mask[relation_labels_idx[0]:relation_labels_idx[-1],
                        relation_labels_idx[0]:relation_labels_idx[-1]] = zeros
            
            attention_mask[0,1:relation_labels_idx[-1]] = np.zeros(shape=(relation_labels_idx[-1]-1,))
            attention_mask[1:relation_labels_idx[-1],0] = np.zeros(shape=(relation_labels_idx[-1]-1,))
            
            for i in range(len(relation_labels_idx)-1):
                label_token_length = relation_labels_idx[i+1]-relation_labels_idx[i]
                ones = np.ones(shape=(label_token_length, label_token_length))
                attention_mask[relation_labels_idx[i]:relation_labels_idx[i+1],
                            relation_labels_idx[i]:relation_labels_idx[i+1]] = ones
            
            zeros = np.zeros(shape=(entity_labels_idx[-1]-entity_labels_idx[0], relation_labels_idx[-1]-relation_labels_idx[0]))
            attention_mask[entity_labels_idx[0]:entity_labels_idx[-1],
                relation_labels_idx[0]:relation_labels_idx[-1]] = zeros
            
            zeros = np.zeros(shape=(relation_labels_idx[-1]-relation_labels_idx[0], entity_labels_idx[-1]-entity_labels_idx[0]))
            attention_mask[relation_labels_idx[0]:relation_labels_idx[-1],
                            entity_labels_idx[0]:entity_labels_idx[-1]] = zeros
                        
            for headtail,relation_list in headtail2relation.items():
                if '|' in headtail:
                    headtail=headtail.split('|')
                else:
                    headtail=[headtail]
                for entity_type in headtail:
                    entity_idx = entity_labels_idx[entity_type_list.index(entity_type)]
                    entity_last_token_idx = entity_labels_idx[entity_type_list.index(entity_type)+1]
                    for relation_type in relation_list:
                        relation_idx = relation_labels_idx[relation_type_list.index(relation_type)]
                        relation_last_token_idx = relation_labels_idx[relation_type_list.index(relation_type)+1]
                        ones = np.ones(shape=(entity_last_token_idx-entity_idx, relation_last_token_idx-relation_idx))
                        attention_mask[entity_idx:entity_last_token_idx,
                            relation_idx:relation_last_token_idx] = ones
            
                        ones = np.ones(shape=(relation_last_token_idx-relation_idx, entity_last_token_idx-entity_idx))
                        attention_mask[relation_idx:relation_last_token_idx,
                                        entity_idx:entity_last_token_idx] = ones
                        
            return attention_mask


    def process_relation_choice(self, choice):
        head_type = []
        tail_type = []
        entity_type = []
        relation_type = []
        headtail2relation={}
        for c in choice:
            if c[0] not in head_type:
                head_type.append(c[0])
            if c[2] not in tail_type:
                tail_type.append(c[2])
                
            if c[0] not in entity_type:
                entity_type.append(c[0])
            if c[2] not in entity_type:
                entity_type.append(c[2])
                
            if c[1] not in relation_type:
                relation_type.append(c[1])
            
            if c[0]+'|'+c[2] not in headtail2relation.keys():
                headtail2relation[c[0]+'|'+c[2]]=[]
            if c[1] not in headtail2relation[c[0]+'|'+c[2]]:
                headtail2relation[c[0]+'|'+c[2]].append(c[1])
                
        return relation_type, entity_type, head_type, tail_type, headtail2relation

    def process_event_choice(self, choice):
        event_type_list=[]
        entity_type_list=[]
        args2event={}
        for event_type,args in choice[0].items():
            if event_type not in event_type_list:
                event_type_list.append(event_type)
            for arg in args:
                if arg not in entity_type_list:
                    entity_type_list.append(arg)
                if arg not in args2event.keys():
                    args2event[arg]=[]
                if event_type not in args2event[arg]:
                    args2event[arg].append(event_type)
        event_type_list.append('触发词与要素')
        return event_type_list, entity_type_list, args2event
    
    def encode_entity(self, text,entity_list,entity_type_list,span_labels,span_labels_mask,span_index_token_list,query_length):
        for entity in entity_list:
            entity_type, entity_idx_list = entity['entity_type'], entity['entity_index']
            for entity_idx in entity_idx_list:
                start_idx, end_idx = self.search_index(
                    entity_idx, text)
                # print(start_idx,end_idx,flush=True)
                if start_idx != None and end_idx != None:
                    start_idx, end_idx = start_idx + \
                        query_length+1, end_idx+query_length+1
                    if start_idx < span_labels.shape[0] and end_idx < span_labels.shape[0]:
                        span_index_token_list.append(start_idx)
                        span_index_token_list.append(end_idx)
                        span_labels[start_idx, end_idx, 0] = 1
                        span_labels_mask[start_idx, end_idx, 0:len(entity_type_list)+1] = np.zeros((len(entity_type_list)+1,))
                        if entity_type in entity_type_list:
                            label = entity_type_list.index(entity_type) + 1  # 加1 是因为entity的 idx从1开始，0是CLS
                            span_labels[start_idx, end_idx, label] = 1
        return span_labels,span_labels_mask,span_index_token_list

    def encode_relation(self, text,spo_list,entity_type_list,relation_type_list,span_labels,span_labels_mask,span_index_token_list,query_length):
        sample_entity_idx_list=[]
        for spo in spo_list:
            for entity_idx_subject in spo['subject']['entity_index']:
                for entity_idx_object in spo['object']['entity_index']:
                    sub_start_idx, sub_end_idx = self.search_index(entity_idx_subject, text)
                    if sub_start_idx !=None and sub_end_idx != None:
                        sub_start_idx, sub_end_idx = sub_start_idx + query_length+1, sub_end_idx + query_length+1
                        sub_label= entity_type_list.index(spo['subject']['entity_type'])+1
                        if sub_start_idx < span_labels.shape[0] and sub_end_idx < span_labels.shape[0]:
                            span_index_token_list.append(sub_start_idx)
                            span_index_token_list.append(sub_end_idx)
                        
                            span_labels[sub_start_idx, sub_end_idx, 0] = 1
                            span_labels[sub_start_idx,sub_end_idx, sub_label] = 1
                            span_labels_mask[sub_start_idx, sub_end_idx, 0:len(entity_type_list)+1] = np.zeros((len(entity_type_list)+1,))
                            if (sub_start_idx, sub_end_idx) not in sample_entity_idx_list:
                                sample_entity_idx_list.append((sub_start_idx, sub_end_idx))

                    ob_start_idx, ob_end_idx = self.search_index(entity_idx_object, text)
                    if ob_start_idx !=None and ob_end_idx != None:
                        ob_start_idx, ob_end_idx = ob_start_idx + query_length+1, ob_end_idx + query_length+1
                        ob_label= entity_type_list.index(spo['object']['entity_type'])+1
                        if ob_start_idx < span_labels.shape[0] and ob_end_idx < span_labels.shape[0]:
                            span_index_token_list.append(ob_start_idx)
                            span_index_token_list.append(ob_end_idx)
                        
                            span_labels[ob_start_idx, ob_end_idx, 0] = 1
                            span_labels[ob_start_idx, ob_end_idx, ob_label] = 1
                            span_labels_mask[ob_start_idx, ob_end_idx, 0:len(entity_type_list)+1] = np.zeros((len(entity_type_list)+1,))
                            if (ob_start_idx, ob_end_idx) not in sample_entity_idx_list:
                                sample_entity_idx_list.append((ob_start_idx, ob_end_idx))


                    if ob_start_idx !=None and ob_end_idx != None and sub_start_idx !=None and sub_end_idx != None:
                        if spo['predicate'] in relation_type_list:
                            rel_label = len(entity_type_list) + relation_type_list.index(spo['predicate'])+1
                            if sub_start_idx < self.max_length and ob_start_idx < self.max_length:
                                span_labels[sub_start_idx,ob_start_idx, rel_label] = 1
                                span_labels_mask[sub_start_idx, ob_start_idx, len(entity_type_list)+1:len(entity_type_list)+len(relation_type_list)+1] = np.zeros((len(relation_type_list),))
                                
                            if sub_end_idx < self.max_length and ob_end_idx < self.max_length:
                                span_labels[sub_end_idx,ob_end_idx, rel_label] = 1
                                span_labels_mask[sub_end_idx, ob_end_idx, len(entity_type_list)+1:len(entity_type_list)+len(relation_type_list)+1] = np.zeros((len(relation_type_list),))  
        
        for head_idx in sample_entity_idx_list:
            for tail_idx in sample_entity_idx_list:
                if head_idx != tail_idx:
                    span_labels_mask[head_idx[0], tail_idx[0], len(entity_type_list)+1:len(entity_type_list)+len(relation_type_list)+1] = np.zeros((len(relation_type_list),))
                    span_labels_mask[head_idx[1], tail_idx[1], len(entity_type_list)+1:len(entity_type_list)+len(relation_type_list)+1] = np.zeros((len(relation_type_list),))
                    
        return span_labels,span_labels_mask,span_index_token_list
        
    def encode_event(self, text,event_list,entity_type_list,event_type_list,span_labels,span_labels_mask,span_index_token_list,query_length,num_labels):
        trigger_args_idx_list=[]
        for event in event_list:
            trigger_idx_list=[]
            args_idx_list=[]
            event_type = event['event_type']
            for entity in event['args']:
                
                entity_type, entity_idx_list = entity['entity_type'], entity['entity_index']
                for entity_idx in entity_idx_list:
                    start_idx, end_idx = self.search_index(
                        entity_idx, text)
                    if start_idx != None and end_idx != None:
                        start_idx, end_idx = start_idx + \
                        query_length+1, end_idx+query_length+1
                        if start_idx < span_labels.shape[0] and end_idx < span_labels.shape[0]:
                            span_index_token_list.append(start_idx)
                            span_index_token_list.append(end_idx)
                        
                            entity_type_label = entity_type_list.index(entity_type) + 1  # 加1 是因为entity的 idx从1开始，0是CLS
                            event_type_label = event_type_list.index(event_type) + len(entity_type_list) + 1  # 加1 是因为entity的 idx从1开始，0是CLS
                            
                            span_labels[start_idx, end_idx, 0] = 1
                            span_labels[start_idx, end_idx, entity_type_label] = 1
                            span_labels[start_idx, end_idx, event_type_label] = 1
                            span_labels_mask[start_idx, end_idx] = np.zeros((num_labels,))
                            
                            if entity_type =='触发词':
                                trigger_idx_list.append((start_idx, end_idx))
                            else:
                                args_idx_list.append((start_idx, end_idx))
                            
            trigger_args_idx_list.append((trigger_idx_list,args_idx_list))
            for trigger_idx in trigger_idx_list:
                for args_idx in args_idx_list:
                    span_labels[trigger_idx[0], args_idx[0], -1] = 1
                    span_labels[trigger_idx[1], args_idx[1], -1] = 1
                    span_labels_mask[trigger_idx[0], args_idx[0], -1] = 0
                    span_labels_mask[trigger_idx[1], args_idx[1], -1] = 0
                    
        for i,(trigger_idx_list1, args_idx_list1) in enumerate(trigger_args_idx_list):
            for j,(trigger_idx_list2, args_idx_list2) in enumerate(trigger_args_idx_list):
                for trigger_idx1 in trigger_idx_list1:
                    for args_idx2 in args_idx_list2:
                        span_labels_mask[trigger_idx1[0], args_idx2[0], -1] = 0
                        span_labels_mask[trigger_idx1[1], args_idx2[1], -1] = 0
                        
                for trigger_idx2 in trigger_idx_list2:
                    for args_idx1 in args_idx_list1:
                        span_labels_mask[trigger_idx2[0], args_idx1[0], -1] = 0
                        span_labels_mask[trigger_idx2[1], args_idx1[1], -1] = 0
        
        return span_labels,span_labels_mask,span_index_token_list

    def encode(self, item,is_predict=False):

        if isinstance(item['choice'][0], list):
            relation_type_list, entity_type_list, _, _, headtail2relation = self.process_relation_choice(item['choice'])
            event_type_list=[]
        elif isinstance(item['choice'][0], dict): # event extraction task
            event_type_list, entity_type_list, args2event = self.process_event_choice(item['choice'])
            relation_type_list = []
        else:
            entity_type_list = item['choice']
            relation_type_list = []
            event_type_list = []

        input_ids = []
        entity_labels_idx = []
        relation_labels_idx = []
        event_labels_idx = []

        sep_ids = [self.tokenizer.sep_token_id]
        subtask_type_ids = self.tokenizer.encode(item['task_type'])

        input_ids = subtask_type_ids
        entity_labels_idx.append(len(input_ids))
        entity_op_ids = self.tokenizer.encode(
            '[unused1]', add_special_tokens=False)[0]
        for c in entity_type_list:
            input_ids = input_ids + \
                [entity_op_ids]+self.tokenizer.encode(c, add_special_tokens=False)
            entity_labels_idx.append(len(input_ids))
        
        
        if relation_type_list !=[]:  #如果不为空，则含有关系类型，添加关系类型
            relation_op_ids = self.tokenizer.encode(
                '[unused2]', add_special_tokens=False)[0]
            input_ids = input_ids + sep_ids
            relation_labels_idx.append(len(input_ids))
            for c in relation_type_list:
                input_ids = input_ids + \
                    [relation_op_ids]+self.tokenizer.encode(c, add_special_tokens=False)
                relation_labels_idx.append(len(input_ids))
        
        
        if event_type_list !=[]:  #如果不为空，则含有事件类型数据
            event_op_ids = self.tokenizer.encode(
                '[unused1]', add_special_tokens=False)[0]
            input_ids = input_ids + sep_ids
            event_labels_idx.append(len(input_ids))
            for c in event_type_list:
                input_ids = input_ids + \
                    [event_op_ids]+self.tokenizer.encode(c, add_special_tokens=False)
                event_labels_idx.append(len(input_ids))
        
        if 'tokens' not in item.keys():
            item['tokens'] = item['text'].split(' ')
        
        if relation_labels_idx!=[]:
            query_length=relation_labels_idx[-1]
        elif event_labels_idx !=[]:
            query_length=event_labels_idx[-1]
        else:
            query_length=entity_labels_idx[-1]
            
        
        encode_dict = self.tokenizer(item['tokens'],
                                     is_split_into_words=True,
                                     max_length=self.max_length-query_length,
                                     truncation=True,
                                     add_special_tokens=False)

        input_ids = input_ids+sep_ids+encode_dict['input_ids']
        word_ids = encode_dict.word_ids()
        

        input_ids = input_ids[:self.args.max_length-1]+sep_ids

        sample_length = len(input_ids)

        attention_mask = [1]*sample_length
        
        if relation_labels_idx!=[]:
            attention_mask = self.get_att_mask(
                attention_mask, entity_labels_idx, relation_labels_idx, entity_type_list, relation_type_list, headtail2relation)
        elif event_labels_idx!=[]:
            attention_mask = self.get_att_mask(
                attention_mask, entity_labels_idx, event_labels_idx, entity_type_list, event_type_list, args2event)
        else:
            attention_mask = self.get_att_mask(
                attention_mask, entity_labels_idx, relation_labels_idx)
            
        if relation_labels_idx !=[]:
            position_ids = self.get_position_ids(
                sample_length, entity_labels_idx, relation_labels_idx)
        else:
            position_ids = self.get_position_ids(
                sample_length, entity_labels_idx, event_labels_idx)

        if relation_type_list !=[]:
            label_token_idx = entity_labels_idx[:-1]+relation_labels_idx[:-1]
            num_labels = len(entity_type_list)+len(relation_type_list)+1 # 加1 是因为entity的 idx从1开始，0是CLS
        elif event_labels_idx !=[]:
            label_token_idx = entity_labels_idx[:-1]+event_labels_idx[:-1]
            num_labels = len(entity_type_list)+len(event_type_list)+1 # 加的第一个1 是因为我们需要一个token来标记trigger和args的关系，第二个是因为entity的 idx从1开始，0是CLS
        else:
            label_token_idx = entity_labels_idx[:-1]
            num_labels = len(entity_type_list)+1 # 加1 是因为entity的 idx从1开始，0是CLS
                   
        span_labels = np.zeros(
            (sample_length, sample_length, num_labels))
        span_mask = True if 'span_mask' in item.keys() and item['span_mask']=='mask' else False
        if not self.args.fast_ex_mode and not span_mask:
            span_labels_mask = np.zeros(
                (sample_length, sample_length, num_labels))
            span_labels_mask[:query_length,:query_length, :] = np.zeros(
                (query_length, query_length, num_labels))-10000
            # span_labels_mask[0, 0, :] = np.zeros((num_labels,))-10000
        else:
            span_labels_mask = np.zeros(
                (sample_length, sample_length, num_labels))-10000
        
            span_labels_mask[query_length:, query_length:, 0] = np.zeros(
                (sample_length-query_length, sample_length-query_length))
            # span_labels_mask[0, 0, :] = np.zeros((num_labels,))
        
        span_index_token_list=[query_length]
        if 'entity_list' in item.keys():
            span_labels,span_labels_mask,span_index_token_list = self.encode_entity(item['text'],
                                                                                    item['entity_list'],
                                                                                    entity_type_list,
                                                                                    span_labels,
                                                                                    span_labels_mask,
                                                                                    span_index_token_list,
                                                                                    query_length)


        if 'spo_list' in item.keys() and relation_type_list !=[]:  # 关系抽取任务
            span_labels,span_labels_mask,span_index_token_list = self.encode_relation(item['text'],
                                                                                      item['spo_list'],
                                                                                      entity_type_list,
                                                                                      relation_type_list,
                                                                                      span_labels,
                                                                                      span_labels_mask,
                                                                                      span_index_token_list,
                                                                                      query_length)
            
        
        if 'event_list' in item.keys():   #针对事件抽取任务
            span_labels,span_labels_mask,span_index_token_list = self.encode_event(item['text'],
                                                                                   item['event_list'],
                                                                                   entity_type_list,
                                                                                   event_type_list,
                                                                                   span_labels,
                                                                                   span_labels_mask,
                                                                                   span_index_token_list,
                                                                                   query_length,
                                                                                   num_labels)
                                    
        token_type_ids = [0]*len(input_ids)
        label_token_idx = [0] + label_token_idx # 加【0】 是因为entity的 idx从1开始，0是CLS
        
        text_token_idx = []
        span_index_token_list=sorted(list(set(span_index_token_list)))
        if is_predict:
            text_token_idx.extend([query_length+idx for idx in range(len(encode_dict['input_ids']))])
        else:
            if self.args.fast_ex_mode and self.args.train:
                text_token_idx.extend(span_index_token_list)
            else:
                text_token_idx.extend([query_length+idx for idx in range(len(encode_dict['input_ids']))])
            
        return {
            "input_ids": torch.tensor(input_ids).long(),
            "token_type_ids": torch.tensor(token_type_ids).long(),
            "attention_mask": torch.tensor(attention_mask).float(),
            "position_ids": torch.tensor(position_ids).long(),
            "span_labels": torch.tensor(span_labels).float(),
            "span_labels_mask": torch.tensor(span_labels_mask).float(),
            "label_token_idx": torch.tensor(label_token_idx).long(),
            "text_token_idx": torch.tensor(text_token_idx).long(),
            "query_length": torch.tensor(query_length).long(),
        }


    def collate_fn(self, batch):
        '''
        Aggregate a batch data.
        batch = [ins1_dict, ins2_dict, ..., insN_dict]
        batch_data = {'sentence':[ins1_sentence, ins2_sentence...], 'input_ids':[ins1_input_ids, ins2_input_ids...], ...}
        '''
        batch_data = {}
        for key in batch[0]:
            batch_data[key] = [example[key] for example in batch]

        batch_data['input_ids'] = nn.utils.rnn.pad_sequence(batch_data['input_ids'],
                                                            batch_first=True,
                                                            padding_value=0)

        batch_size, batch_max_length = batch_data['input_ids'].shape

        batch_data['label_token_idx'] = nn.utils.rnn.pad_sequence(batch_data['label_token_idx'],
                                                                  batch_first=True,
                                                                  padding_value=0)
        batch_data['text_token_idx'] = nn.utils.rnn.pad_sequence(batch_data['text_token_idx'],
                                                                  batch_first=True,
                                                                  padding_value=0)
        batch_data['query_length'] =  torch.tensor(batch_data['query_length']).long()
        batch_size, batch_max_labels = batch_data['label_token_idx'].shape

        for k, v in batch_data.items():
            if k in ['input_ids', 'label_token_idx','text_token_idx','query_length']:
                continue
            if k in ['token_type_ids', 'position_ids']:
                batch_data[k] = nn.utils.rnn.pad_sequence(v,
                                                          batch_first=True,
                                                          padding_value=0)
            elif k == 'attention_mask':
                attention_mask = torch.zeros(
                    (batch_size, batch_max_length, batch_max_length))
                for i, att in enumerate(v):
                    sample_length, _ = att.shape
                    attention_mask[i, :sample_length, :sample_length] = att
                batch_data[k] = attention_mask
            elif k == 'span_labels':
                span = torch.zeros(
                    (batch_size, batch_max_length, batch_max_length, batch_max_labels))
                for i, s in enumerate(v):
                    sample_length, _, sample_num_labels = s.shape
                    span[i, :sample_length, :sample_length,
                         :sample_num_labels] = s
                batch_data[k] = span
            elif k == 'span_labels_mask':
                span = torch.zeros(
                    (batch_size, batch_max_length, batch_max_length, batch_max_labels))-10000
                for i, s in enumerate(v):
                    sample_length, _, sample_num_labels = s.shape
                    span[i, :sample_length, :sample_length,
                         :sample_num_labels] = s
                batch_data[k] = span
        return batch_data
    
    
class UniEXDataset(Dataset):
    def __init__(self, data, tokenizer, args, data_encode):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.data = data
        self.args = args
        self.data_encode = data_encode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data_encode.encode(self.data[index])


class UniEXDataModel(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('TASK NAME DataModel')
        parser.add_argument('--num_workers', default=8, type=int)
        parser.add_argument('--batchsize', default=16, type=int)
        parser.add_argument('--max_length', default=512, type=int)

        return parent_args

    def __init__(self, train_data, dev_data, tokenizer, args):
        super().__init__()
        self.args = args
        self.data_encode = UniEXDataEncode(tokenizer, args)
        self.train_data = UniEXDataset(train_data, tokenizer, args, self.data_encode)
        self.valid_data = UniEXDataset(dev_data, tokenizer, args, self.data_encode)
        

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, collate_fn=self.data_encode.collate_fn ,num_workers=self.args.num_workers, batch_size=self.args.batchsize, pin_memory=False)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, collate_fn=self.data_encode.collate_fn, num_workers=1,  batch_size=self.args.batchsize, pin_memory=False)

class MultilabelCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_true = y_true.float()
        y_pred = torch.mul((1.0 - torch.mul(y_true, 2.0)), y_pred)
        y_pred_neg = y_pred - torch.mul(y_true, 1e12)
        y_pred_pos = y_pred - torch.mul(1.0 - y_true, 1e12)
        zeros = torch.zeros_like(y_pred[..., :1]) 
        y_pred_neg = torch.cat([y_pred_neg, zeros], axis=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], axis=-1)
        neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
        pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
        loss = torch.mean(neg_loss + pos_loss)
        return loss
 

class Triaffine(nn.Module):
    def __init__(self, triaffine_hidden_size):
        super().__init__()

        self.triaffine_hidden_size = triaffine_hidden_size
        self.weight = torch.nn.Parameter(torch.zeros(
            triaffine_hidden_size, triaffine_hidden_size, triaffine_hidden_size))
        torch.nn.init.normal_(self.weight, mean=0, std=0.1)
        
    def forward(self, start_logits, end_logits, cls_logits):
        span_logits = torch.einsum(
            'bxi,ioj,byj->bxyo', start_logits, self.weight, end_logits)
        span_logits = torch.einsum(
            'bxyo,bzo->bxyz', span_logits, cls_logits)
        return span_logits


class MLPLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(
            in_features=input_size, out_features=output_size), torch.nn.GELU())

    def forward(self, hidden_state):
        return self.mlp(hidden_state)


class UniEXBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.config = config
        self.mlp_start = MLPLayer(
            self.config.hidden_size, self.config.triaffine_hidden_size)
        self.mlp_end = MLPLayer(self.config.hidden_size,
                                self.config.triaffine_hidden_size)
        self.mlp_cls = MLPLayer(self.config.hidden_size,
                                self.config.triaffine_hidden_size)

        self.triaffine = Triaffine(self.config.triaffine_hidden_size)

        self.loss_softmax = MultilabelCrossEntropy()
        self.loss_sigmoid = torch.nn.BCEWithLogitsLoss()

    def span_gather(self,span_labels,text_token_idx):
        """从表为 【seq_len，seq_len】 的 span_labels 提取 只需要计算 loss 的 token 的labels，提取之后，size ->【token_len, token_len】
        """
        try:
            batch_size,seq_len,_,num_labels=span_labels.shape
            _,text_len=text_token_idx.shape
            e=torch.arange(seq_len)*seq_len
            e=e.to(span_labels.device)
            e=e.unsqueeze(0).unsqueeze(-1).repeat(batch_size,1, text_len)
            e=e.gather(1, text_token_idx.unsqueeze(-1).repeat(1, 1, text_len))
            text_token_idx = text_token_idx.unsqueeze(1).repeat(1, text_len, 1)
            text_token_idx = text_token_idx+e
            text_token_idx=text_token_idx.reshape(-1,text_len*text_len)
            span_labels=span_labels.reshape(-1,seq_len*seq_len,num_labels)
            span_labels=span_labels.gather(1, text_token_idx.unsqueeze(-1).repeat(1, 1, num_labels))
            span_labels=span_labels.reshape(-1,text_len,text_len,num_labels)
        except:
            print(span_labels.shape)
            print(text_token_idx.shape)
                
        return span_labels
    
    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                query_length=None,
                position_ids=None,
                span_labels=None,
                span_labels_mask=None,
                label_token_idx=None,
                text_token_idx=None,
                fast_ex_mode=False,
                task_type_list=None,
                threshold=0.5):

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            token_type_ids=token_type_ids,
                            output_hidden_states=True)  # (bsz, seq, dim)
        
        hidden_states = outputs[0]
        batch_size, seq_len, hidden_size = hidden_states.shape
        if span_labels_mask != None:     
            start_logits = self.mlp_start(hidden_states)
            end_logits = self.mlp_end(hidden_states)
            cls_logits = hidden_states.gather(
                1, label_token_idx.unsqueeze(-1).repeat(1, 1, hidden_size))
            cls_logits = self.mlp_cls(cls_logits)
            
            span_logits = self.triaffine(start_logits, end_logits, cls_logits[:,[0],:])
            span_logits = span_logits + span_labels_mask[:,:,:,[0]]
            index_loss_sigmoid = self.loss_sigmoid(span_logits, span_labels[:,:,:,[0]])

            start_logits = start_logits.gather(
                1, text_token_idx.unsqueeze(-1).repeat(1, 1, self.config.triaffine_hidden_size))
            end_logits = end_logits.gather(
                1, text_token_idx.unsqueeze(-1).repeat(1, 1, self.config.triaffine_hidden_size))
            
            span_logits = self.triaffine(start_logits, end_logits, cls_logits[:,1:,:])
            span_labels = self.span_gather(span_labels[:,:,:,1:],text_token_idx)
            span_labels_mask = self.span_gather(span_labels_mask[:,:,:,1:],text_token_idx)
            span_logits = span_logits + span_labels_mask
            span_loss_sigmoid = self.loss_sigmoid(span_logits, span_labels)
            all_loss = 100000*span_loss_sigmoid + 100000*index_loss_sigmoid 
            return all_loss, span_logits, span_labels
        else:
            if not fast_ex_mode:
                text_logits = hidden_states.gather(
                1, text_token_idx.unsqueeze(-1).repeat(1, 1, hidden_size))
                start_logits = self.mlp_start(text_logits)
                end_logits = self.mlp_end(text_logits)
                cls_logits = hidden_states.gather(
                    1, label_token_idx.unsqueeze(-1).repeat(1, 1, hidden_size))
                cls_logits = self.mlp_cls(cls_logits)
                span_logits = self.triaffine(start_logits, end_logits, cls_logits)
                span_logits = torch.sigmoid(span_logits)
                return span_logits
            
            else:
                text_logits = hidden_states.gather(
                    1, text_token_idx.unsqueeze(-1).repeat(1, 1, hidden_size))
                start_logits = self.mlp_start(text_logits)
                end_logits = self.mlp_end(text_logits)

                cls_logits = hidden_states.gather(
                    1, label_token_idx.unsqueeze(-1).repeat(1, 1, hidden_size))
                
                cls_logits = self.mlp_cls(cls_logits)
                
                index_logits=cls_logits[:,:1,:]
                type_logits=cls_logits[:,1:,:]
                
                span_index_logits = self.triaffine(start_logits, end_logits, index_logits).squeeze(-1)
                span_index_logits = torch.sigmoid(span_index_logits)
                token_index = torch.zeros(size=(batch_size,seq_len)).to(input_ids.device)
                max_num_index = 0
                span_index_list=[]
                span_index_labels=self.span_gather(span_labels[:,:,:,[0]],text_token_idx).squeeze(-1)
                for idx in range(batch_size):
                    if task_type_list is not None and task_type_list[idx] in ['分类任务']:
                        span_index = span_index_labels[idx] > threshold
                    else:
                        span_index = span_index_logits[idx] > threshold
                    span_index = span_index.nonzero()
                    span_index_list.append(span_index)
                    span_index = span_index.reshape((-1,))
                    span_index = torch.unique(span_index,sorted=True)

                    num_span_index = span_index.shape[0]
                    token_index[idx,:num_span_index]=span_index
                    max_num_index = num_span_index if max_num_index < num_span_index else max_num_index

                token_index = token_index[:,:max_num_index].long()
                start_logits = start_logits.gather(
                    1, token_index.unsqueeze(-1).repeat(1, 1, self.config.triaffine_hidden_size))
                end_logits = end_logits.gather(
                    1, token_index.unsqueeze(-1).repeat(1, 1, self.config.triaffine_hidden_size))

                span_type_logits = self.triaffine(start_logits, end_logits, type_logits)
                span_type_logits = torch.sigmoid(span_type_logits)
                return span_index_logits, span_type_logits, span_index_list, token_index
        

class UniEXLitModel(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')

        parser.add_argument('--learning_rate', default=1e-5, type=float)
        parser.add_argument('--weight_decay', default=0.1, type=float)
        parser.add_argument('--warmup', default=0.01, type=float)

        return parent_args

    def __init__(self, args, dev_data=[],test_data=[], num_data=1):
        super().__init__()
        self.args = args
        self.num_data = num_data
        self.config=AutoConfig.from_pretrained(args.pretrained_model_path)
        self.model = UniEXBertModel.from_pretrained(args.pretrained_model_path)
            
        self.computacc=ComputAcc(args)
        self.dev_data=dev_data
        self.test_data=test_data
        
        self.corr_count=0
        self.gold_count=0
        self.pred_count=0

    def setup(self, stage) -> None:
        if stage == 'fit':
            num_gpus = self.trainer.num_devices if self.trainer.num_devices is not None else 0
            self.total_step = int(self.trainer.max_epochs * self.num_data /
                                  (max(1, num_gpus) * self.trainer.accumulate_grad_batches))
            print('Total training step:', self.total_step)

    def training_step(self, batch, batch_idx):
        loss, span_logits, span_labels = self.model(**batch)
        f1, recall, precise, _, _, _ = self.comput_metrix(
            span_logits, span_labels)

        self.log('train_loss', loss)
        self.log('train_f1', f1)
        self.log('train_recall', recall)
        self.log('train_precise', precise)
        return loss
    
    def training_epoch_end(self,batch):
        f1, recall, precise=self.computacc.predict(self.dev_data,self.model)
        
        self.log('val_f1', f1)
        self.log('val_recall', recall)
        self.log('val_precise', precise)
        
        if self.test_data!=[]:
            f1, recall, precise=self.computacc.predict(self.test_data,self.model)
        else:
            f1, recall, precise=0,0,0
            
        self.log('test_f1', f1)
        self.log('test_recall', recall)
        self.log('test_precise', precise)
        gc.collect()
        
        
    def configure_optimizers(self):

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        paras = list(
            filter(lambda p: p[1].requires_grad, self.named_parameters()))

        paras = [{
            'params':
            [p for n, p in paras if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay
        }, {
            'params': [p for n, p in paras if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }]

        optimizer = torch.optim.AdamW(paras, lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.total_step * self.args.warmup),
            self.total_step)

        return [{
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }]


    def comput_metrix(self, logits, labels):
        logits = torch.nn.functional.sigmoid(logits)  # [b,s,s]
        ones = torch.ones_like(logits)
        zero = torch.zeros_like(logits)
        logits = torch.where(logits < 0.5, zero, ones)
        y_pred = logits.reshape(shape=(-1,))
        y_true = labels.reshape(shape=(-1,))
        corr = torch.eq(y_pred, y_true).float()
        corr = torch.multiply(y_true, corr)
        if torch.sum(y_true.float()) <= 0:
            recall = 0
        else:
            recall = torch.sum(corr.float())/(torch.sum(y_true.float()))
        if torch.sum(y_pred.float()) <= 0:
            precise = 0
        else:
            precise = torch.sum(corr.float())/(torch.sum(y_pred.float()))
        if recall+precise <= 0:
            f1 = 0
        else:
            f1 = 2*recall*precise/(recall+precise)
        return f1, recall, precise, torch.sum(corr.float()), torch.sum(y_true.float()), torch.sum(y_pred.float())


class TaskModelCheckpoint:
    @staticmethod
    def add_argparse_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')

        parser.add_argument('--monitor', default='train_loss', type=str)
        parser.add_argument('--mode', default='min', type=str)
        parser.add_argument('--dirpath', default='./log/', type=str)
        parser.add_argument(
            '--filename', default='model-{epoch:02d}-{train_loss:.4f}', type=str)

        parser.add_argument('--save_top_k', default=3, type=float)
        parser.add_argument('--every_n_epochs', default=1, type=float)
        parser.add_argument('--every_n_train_steps', default=100, type=float)
        parser.add_argument('--save_weights_only', default=True, type=bool)
        return parent_args

    def __init__(self, args):
        self.callbacks = ModelCheckpoint(monitor=args.monitor,
                                         save_top_k=args.save_top_k,
                                         mode=args.mode,
                                         save_last=True,
                                         every_n_epochs=args.every_n_epochs,
                                         save_weights_only=args.save_weights_only,
                                         dirpath=args.dirpath,
                                         filename=args.filename)


class OffsetMapping:
    def __init__(self):
        self._do_lower_case = True
    @staticmethod
    def stem(token):
            if token[:2] == '##':
                return token[2:]
            else:
                return token
    @staticmethod
    def _is_control(ch):
            return unicodedata.category(ch) in ('Cc', 'Cf')
    @staticmethod
    def _is_special(ch):
            return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def rematch(self, text, tokens):
        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([offset+oi for oi in range(len(token))])
                offset+=1
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end
        return token_mapping
    

class FastExtractModel:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.fast_ex_mode = True if args.fast_ex_mode else False
        self.data_encode = UniEXDataEncode(tokenizer, args)
        self.offset_mapping_model = OffsetMapping()

    def extract_index(self, span_logits, sample_length, split_value=0.5):
        result = []
        for i in range(sample_length):
            for j in range(i, sample_length):
                if span_logits[i, j] > split_value:
                    result.append((i, j, span_logits[i, j]))
        return result
    
    def extract_entity(self, text, entity_idx, text_start_id, offset_mapping):
        start_split=offset_mapping[entity_idx[0]-text_start_id] if entity_idx[0]-text_start_id<len(offset_mapping) and entity_idx[0]-text_start_id>=0 else []
        end_split=offset_mapping[entity_idx[1]-text_start_id] if entity_idx[1]-text_start_id<len(offset_mapping) and entity_idx[1]-text_start_id>=0 else []
        if start_split!=[] and end_split!=[]:
            entity = text[start_split[0]:end_split[-1]+1]
            return entity,start_split[0],end_split[-1]
        else:
            return '',0,0

    def extract(self, batch_data, model):

        batch = [self.data_encode.encode(
            sample,is_predict=True) for sample in batch_data]
        batch = self.data_encode.collate_fn(batch)
        new_batch = {}
        for k, v in batch.items():
            if k not in ['span_labels_mask']:
                new_batch[k]=v.cuda()
        task_type_list=[item['task_type'] for item in batch_data]
        span_index_logits, span_type_logits, span_index_list, token_index = model(**new_batch,fast_ex_mode=self.fast_ex_mode,task_type_list=task_type_list,threshold=self.args.threshold_index)

        token_index=token_index.cpu().detach().numpy()
        span_type_logits=span_type_logits.cpu().detach().numpy()
        
        query_len = 1
        for i, item in enumerate(batch_data):
            if 'tokens' in batch_data[i].keys():
                del batch_data[i]['tokens']
            if isinstance(item['choice'][0], list):
                relation_type_list, entity_type_list, head_type, tail_type, headtail2relation = self.data_encode.process_relation_choice(item['choice'])
            else:
                entity_type_list = item['choice']
                relation_type_list = []
            token_index2index={v:idx for idx,v in enumerate(token_index[i])}
            tokens = self.tokenizer.tokenize(item['text'])


            offset_mapping = self.offset_mapping_model.rematch(item['text'],tokens)
            
            sample_length = min(query_len + len(tokens), self.args.max_length - 1)
            encode_dict = self.tokenizer(item['tokens'],
                                     is_split_into_words=True,
                                     max_length=self.args.max_length,
                                     truncation=True,
                                     add_special_tokens=False)
            word_ids = encode_dict.word_ids()


            if item['task_type'] in ['实体识别'] and isinstance(item['choice'][0], str):
                """
                实体抽取解码过程如下：
                1、抽取实体的位置，span_index_list
                2、遍历span_index_list，识别每个span的类别
                """
                entity_logits = span_type_logits[i]
                entity_list = []
                for entity_idx in span_index_list[i].cpu().detach().numpy():
  
                    entity_start = token_index2index[entity_idx[0]]
                    entity_end = token_index2index[entity_idx[1]]

                    entity_type_idx = np.argmax(
                                entity_logits[entity_start, entity_end])
                    entity_type_score = entity_logits[entity_start, entity_end, entity_type_idx]
                    if entity_type_score>self.args.threshold_entity*0:
                        entity_type = entity_type_list[entity_type_idx]
                        entity,start_idx,end_idx = self.extract_entity(item['text'],[entity_idx[0], entity_idx[1]],query_len,offset_mapping)
                        entity = {
                            'entity_text': entity,
                            'entity_type': entity_type,
                            'type_score': float(entity_type_score),
                            'entity_index': [[start_idx,end_idx]]
                        }
                        if entity not in entity_list:
                            entity_list.append(entity)
                            
                batch_data[i]['entity_list'] = entity_list

            if item['task_type'] in ['分类任务'] and isinstance(item['choice'][0], str):
                entity_logits = span_type_logits[i]
                entity_list = []
                for entity_idx in span_index_list[i].cpu().detach().numpy():
  
                    entity_start = token_index2index[entity_idx[0]]
                    entity_end = token_index2index[entity_idx[1]]

                    entity_type_idx = np.argmax(
                                entity_logits[entity_start, entity_end])
                    entity_type_score = entity_logits[entity_start, entity_end, entity_type_idx]
                    if entity_type_score>self.args.threshold_entity*0:
                        entity_type = entity_type_list[entity_type_idx]
                        entity,start_idx,end_idx = self.extract_entity(item['text'],[entity_idx[0], entity_idx[1]],query_len,offset_mapping)
                        entity = {
                            'entity_text': entity,
                            'entity_type': entity_type,
                            'type_score': float(entity_type_score),
                            'entity_index': [[start_idx,end_idx]]
                        }
                        if entity not in entity_list:
                            entity_list.append(entity)
                            
                batch_data[i]['entity_list'] = entity_list


            elif item['task_type'] in ['关系抽取','指代消解']:
                """
                实体抽取解码过程如下：
                1、抽取实体的位置，span_index_list
                2、遍历span_index_list，识别每个span的类别，并确定是head还是tail的实体。得到entity_idx_type_list_head，entity_idx_type_list_tail
                3、遍历head和tail，判断一对<head,tail>是否构成一对关系。
                """
                assert isinstance(item['choice'][0], list)
                relation_type_list, entity_type_list, head_type, tail_type, headtail2relation = self.data_encode.process_relation_choice(item['choice'])
                
                entity_logits = span_type_logits[i][:, :, :len(entity_type_list)]  # 3 
                rel_logits = span_type_logits[i][:, :, len(entity_type_list):len(entity_type_list)+len(relation_type_list)]   # 2


                entity_idx_list = span_index_list[i].cpu().detach().numpy()
                if entity_idx_list.shape[0]>sample_length/2:
                    batch_data[i]['entity_list'] = []
                    batch_data[i]['spo_list'] = []
                else:
                    entity_list = []
                    entity_idx_type_list_head = []  # head 
                    entity_idx_type_list_tail = []   #尾
                    for entity_idx in entity_idx_list:
                        entity_start = token_index2index[entity_idx[0]]
                        entity_end = token_index2index[entity_idx[1]]
                    
                        entity_type_idx = np.argmax(
                                    entity_logits[entity_start, entity_end])
                        entity_type_score = entity_logits[entity_start, entity_end,entity_type_idx]
                        entity_type = entity_type_list[entity_type_idx]
                        if entity_type_score>self.args.threshold_entity*0:
                            if entity_type in head_type:
                                entity_idx_type_list_head.append((entity_idx[0], entity_idx[1], entity_type, entity_type_score))
                            if entity_type in tail_type:
                                entity_idx_type_list_tail.append((entity_idx[0], entity_idx[1], entity_type, entity_type_score))
                    spo_list = []
                    entity_list = []
                    for entity_head in entity_idx_type_list_head:
                        for entity_tail in entity_idx_type_list_tail:
                            subject_type = entity_head[2]
                            object_type = entity_tail[2]
                            if subject_type + '|' + object_type not in headtail2relation.keys():
                                continue
                            so_rel=headtail2relation[subject_type + '|' + object_type]
                            so_rel_idx=[relation_type_list.index(r) for r in so_rel]
                            
                            predicate=None
                            if len(so_rel_idx)>=1:
                                so_rel_logits = rel_logits[:, :, so_rel_idx]
                                hh = np.argmax(
                                    so_rel_logits[token_index2index[entity_head[0]], token_index2index[entity_tail[0]]])
                                tt = np.argmax(
                                    so_rel_logits[token_index2index[entity_head[1]], token_index2index[entity_tail[1]]])
                            
                                hh_score=so_rel_logits[token_index2index[entity_head[0]], token_index2index[entity_tail[0]], hh]
                                tt_score=so_rel_logits[token_index2index[entity_head[1]], token_index2index[entity_tail[1]], tt]
                                
                                if hh_score>tt_score:
                                    idx=hh
                                    ht_score = hh_score
                                else:
                                    idx=tt
                                    ht_score = tt_score
                                
                                if ht_score>self.args.threshold_relation:
                                    predicate = so_rel[idx]
                                    predicate_score = ht_score
                        
                            entity_subject,start_idx,end_idx = self.extract_entity(item['text'], [entity_head[0], entity_head[1]],query_len,offset_mapping)
                            subject_dict = {
                                'entity_text': entity_subject, 
                                'entity_type': subject_type, 
                                'score': float(entity_head[3]), 
                                'entity_index': [[start_idx,end_idx]]}

                            entity_object,start_idx,end_idx = self.extract_entity(item['text'], [entity_tail[0], entity_tail[1]],query_len,offset_mapping)
                            object_dict = {
                                'entity_text': entity_object, 
                                'entity_type': object_type, 
                                'score': float(entity_tail[3]), 
                                'entity_index': [[start_idx,end_idx]]}
                            
                            entity_list.append(subject_dict)
                            entity_list.append(object_dict)
                            
                            if predicate != None:
                                spo = {
                                    'predicate': predicate,
                                    'score': predicate_score,
                                    'subject': subject_dict,
                                    'object': object_dict,
                                }
                                spo_list.append(spo)
                            
                    batch_data[i]['entity_list'] = entity_list
                    batch_data[i]['spo_list'] = spo_list
                              
            elif item['task_type'] in ['事件抽取'] and isinstance(item['choice'][0], dict):
                """
                实体抽取解码过程如下：
                1、抽取实体的位置，span_index_list
                2、遍历span_index_list，同时识别该 span 的 event_type 和 entity_type 。取event_type_score+entity_type_score对应的类别
                3、遍历head和tail，判断一对<head,tail>是否构成一对关系。
                """
                event_type_list, entity_type_list, args2event = self.data_encode.process_event_choice(item['choice'])
                
                event_args_type_list=[]
                for at,et_list in args2event.items():
                    for et in et_list:
                        if (et,at) not in event_args_type_list:
                            event_args_type_list.append((et,at))
                
                entity_logits = span_type_logits[i][:, :, :len(entity_type_list)]
                event_logits = span_type_logits[i][:, :, len(entity_type_list): len(entity_type_list)+len(event_type_list)]
                trigger_args_logits = span_type_logits[i][:,:,-1]
                
                entity_logits = np.tile(entity_logits[:, :, np.newaxis, :],[1,1,len(event_type_list),1])
                event_logits = np.tile(event_logits[:, :, :, np.newaxis],[1,1,1,len(entity_type_list)])
                event_entity_logits = (event_logits+entity_logits)/2 
                seq_len,seq_len,etl,atl=event_entity_logits.shape
                for ei,et in enumerate(event_type_list):
                    for ai,at in enumerate(entity_type_list):
                        if (et,at) not in event_args_type_list:
                            event_entity_logits[:,:,ei,ai]=np.zeros((seq_len,seq_len))
                
                entity_idx_list = span_index_list[i].cpu().detach().numpy()
                pred_event_type_list = []
                args_list=[]
                trigger_list=[]
                
                for entity_idx in entity_idx_list:

                    entity_start = token_index2index[entity_idx[0]]
                    entity_end = token_index2index[entity_idx[1]]
                    
                    event_entity_type_idx = np.unravel_index(np.argmax(event_entity_logits[entity_start, entity_end], axis=None), event_entity_logits[entity_start, entity_end].shape)

                    entity_type_score = entity_logits[entity_start, entity_end,event_entity_type_idx[0],event_entity_type_idx[1]]
                    entity_type = entity_type_list[event_entity_type_idx[1]]
                    event_type = event_type_list[event_entity_type_idx[0]]
                    
                    entity,start_idx,end_idx = self.extract_entity(item['text'],[entity_idx[0], entity_idx[1]],query_len,offset_mapping)
                    if entity != '':
                        entity = {
                            'entity_text': entity,
                            'entity_type': entity_type,
                            'entity_score': float(entity_type_score),
                            'entity_index': [[start_idx,end_idx]]
                        }
                        event={}
                        event['event_type']=event_type
                        event['args']= [entity]
                        if event_type not in pred_event_type_list:
                            pred_event_type_list.append(event_type)
                        
                        if entity_type == '触发词':
                            trigger_list.append((event,entity_start, entity_end))
                        else:
                            args_list.append((event,entity_start, entity_end))
                
                if len(trigger_list)+len(args_list)>sample_length/4:
                    batch_data[i]['event_list'] = []
                    continue      
                
                event_list=[]
                for event_type in pred_event_type_list:
                    tmp_e_list=[]
                    trigger_idx_list=[]
                    for e in trigger_list:
                        if e[0]['event_type']==event_type:
                            tmp={}
                            tmp['event_type']=event_type
                            tmp['args']=e[0]['args']
                            tmp_e_list.append(tmp)
                            trigger_idx_list.append(e)
                            
                    if tmp_e_list==[]:
                        tmp={}
                        tmp['event_type']=event_type
                        tmp['args']=[]
                        tmp_e_list.append(tmp)
                                              
                    for e in args_list:
                        if e[0]['event_type']==event_type:
                            if trigger_idx_list==[]:
                                tmp_e_list[0]['args'].extend(e[0]['args'])
                            else:
                                scores=[]
                                for t in trigger_idx_list:
                                    score=trigger_args_logits[e[1],t[1]]+trigger_args_logits[e[2],t[2]]
                                    scores.append(score)
                                et=scores.index(max(scores))
                                tmp_e_list[et]['args'].extend(e[0]['args'])
                    event_list.extend(tmp_e_list)
                batch_data[i]['event_list'] = event_list
                
        return batch_data


class ExtractModel:
    
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.fast_ex_mode = True if args.fast_ex_mode else False
        self.data_encode= UniEXDataEncode(tokenizer, args)
        self.offset_mapping_model = OffsetMapping()


    def extract_index(self, span_logits, sample_length, split_value=0.5):
        result = []
        for i in range(span_logits.shape[0]):
            for j in range(i, span_logits.shape[1]):
                c = np.argmax(span_logits[i, j])
                # for c in range(span_logits.shape[2]):
                if span_logits[i, j, c] > split_value:
                    result.append((i, j, c, span_logits[i, j, c]))
        return result
    
    
    def extract_entity(self, text,entity_idx,text_start_id,offset_mapping):
        start_split=offset_mapping[entity_idx[0]-text_start_id] if entity_idx[0]-text_start_id<len(offset_mapping) and entity_idx[0]-text_start_id>=0 else []
        end_split=offset_mapping[entity_idx[1]-text_start_id] if entity_idx[1]-text_start_id<len(offset_mapping) and entity_idx[1]-text_start_id>=0 else []
        if start_split!=[] and end_split!=[]:
            entity = text[start_split[0]:end_split[-1]+1]
            return entity,start_split[0],end_split[-1]
        else:
            return '',0,0


    def extract(self, batch_data, model):

        batch = [self.data_encode.encode(
            sample,is_predict=True) for sample in batch_data]
        batch = self.data_encode.collate_fn(batch)
        new_batch = {}
        for k, v in batch.items():
            if k not in ['span_labels','span_labels_mask']:
                new_batch[k]=v.to(model.device)
    
        span_logits = model(**new_batch, fast_ex_mode=self.fast_ex_mode)

        span_logits = span_logits.cpu().detach().numpy()
        span_logits = span_logits[:,:,:,1:]
        query_len = 1
        for i, item in enumerate(batch_data):
            if 'tokens' in batch_data[i].keys():
                del batch_data[i]['tokens']
            if isinstance(item['choice'][0], list):
                relation_type_list, entity_type_list, head_type, tail_type, headtail2relation = self.data_encode.process_relation_choice(item['choice'])
            else:
                entity_type_list = item['choice']
                relation_type_list = []
            
            tokens = self.tokenizer.tokenize(item['text'])
            offset_mapping = self.offset_mapping_model.rematch(item['text'],tokens)
            sample_length = min(query_len + len(tokens), self.args.max_length - 1)
            
            if item['task_type'] in ['实体识别'] and isinstance(item['choice'][0], str):
                
                entity_idx_type_list = self.extract_index(
                    span_logits[i], sample_length)
                entity_list = []
                for entity_idx_type in entity_idx_type_list:
                    entity_start, entity_end, entity_type, score = entity_idx_type
                    entity_type = entity_type_list[entity_type]
                    entity,start_idx,end_idx = self.extract_entity(item['text'],[entity_start, entity_end],query_len,offset_mapping)
                    if entity != '':
                        entity = {
                            'entity_text': entity,
                            'entity_type': entity_type,
                            'score': float(score),
                            'entity_index': [[start_idx,end_idx]]
                        }
                        if entity not in entity_list:
                            entity_list.append(entity)
                batch_data[i]['entity_list'] = entity_list
            
            elif item['task_type'] in ['抽取式阅读理解']:
                entity_list = []
                for c in range(len(item['choice'])):
                    logits = span_logits[i]
                    max_index = np.unravel_index(
                        np.argmax(logits, axis=None), logits.shape)
                    
                    if logits[max_index] < self.args.threshold_entity:
                        entity = {
                            'entity_name': '',
                            'entity_type': item['choice'][c],
                            'score': float(logits[max_index]),
                            'entity_idx': [[]]
                        }
                        if entity not in entity_list:
                            entity_list.append(entity)
                            
                    else:
                        entity,start_idx,end_idx = self.extract_entity(item['text'],max_index, query_len, offset_mapping)
                        entity = {
                            'entity_name': entity,
                            'entity_type': item['choice'][c],
                            'score': float(logits[max_index]),
                            'entity_idx': [[start_idx,end_idx]]
                        }
                        if entity not in entity_list:
                            entity_list.append(entity)
                batch_data[i]['entity_list'] = entity_list
                
            elif item['task_type'] in ['关系抽取','指代消解'] and isinstance(item['choice'][0], list):
                assert isinstance(item['choice'][0], list)
                relation_type_list, entity_type_list, head_type, tail_type, headtail2relation = self.data_encode.process_relation_choice(item['choice'])
                head_type_idx=[entity_type_list.index(et) for et in head_type]
                tail_type_idx=[entity_type_list.index(et) for et in tail_type]
                
                head_logits = span_logits[i][:, :, head_type_idx]
                tail_logits = span_logits[i][:, :, tail_type_idx]
                rel_logits = span_logits[i][:, :, len(entity_type_list):len(entity_type_list)+len(relation_type_list)]

                entity_idx_type_list_head = self.extract_index(
                    head_logits, sample_length, split_value=self.args.threshold_entity)
                entity_idx_type_list_tail = self.extract_index(
                    tail_logits, sample_length, split_value=self.args.threshold_entity)
                
                if len(entity_idx_type_list_head)+len(entity_idx_type_list_tail)>sample_length/2:
                    batch_data[i]['entity_list'] = []
                    batch_data[i]['spo_list'] = []
                else:
                    spo_list = []
                    entity_list = []
                    for entity_head in entity_idx_type_list_head:
                        for entity_tail in entity_idx_type_list_tail:
                            
                            subject_type = head_type[entity_head[2]]
                            object_type = tail_type[entity_tail[2]]
                            
                            entity_subject,start_idx,end_idx = self.extract_entity(item['text'], [entity_head[0], entity_head[1]], query_len, offset_mapping)
                            subject_dict = {
                                'entity_text': entity_subject, 
                                'entity_type': subject_type, 
                                'score': float(entity_head[3]), 
                                'entity_index': [[start_idx,end_idx]]}
                            entity_list.append(subject_dict)
                            entity_object,start_idx,end_idx = self.extract_entity(item['text'], [entity_tail[0], entity_tail[1]], query_len, offset_mapping)
                            object_dict = {
                                'entity_text': entity_object, 
                                'entity_type': object_type, 
                                'score': float(entity_tail[3]), 
                                'entity_index': [[start_idx,end_idx]]
                                }
                            entity_list.append(object_dict)
                            
                            if subject_type + '|' + object_type not in headtail2relation.keys():
                                continue
                            
                            so_rel=headtail2relation[subject_type + '|' + object_type]
                            so_rel_idx=[relation_type_list.index(r) for r in so_rel]
                        
                            predicate=None
                            if len(so_rel_idx)>=1:
                                so_rel_logits = rel_logits[:, :, so_rel_idx]
                                hh = np.argmax(
                                    so_rel_logits[entity_head[0], entity_tail[0]])
                                tt = np.argmax(
                                    so_rel_logits[entity_head[1], entity_tail[1]])
                                
                                hh_score=so_rel_logits[entity_head[0], entity_tail[0], hh]+so_rel_logits[entity_head[1], entity_tail[1], hh]
                                tt_score=so_rel_logits[entity_head[0], entity_tail[0], tt]+so_rel_logits[entity_head[1], entity_tail[1], tt]
                                
                                idx=hh if hh_score>tt_score else tt
                                
                                ht_score = so_rel_logits[entity_head[0], entity_tail[0], idx]+so_rel_logits[entity_head[1], entity_tail[1], idx]
        
                                if ht_score/2>self.args.threshold_relation:
                                    predicate=so_rel[idx]
                                    predicate_score=ht_score/2
                            
                            if predicate != None:
                                if entity_subject != '' and entity_object != '':
                                    spo = {
                                        'predicate': predicate,
                                        'score': float(predicate_score),
                                        'subject': subject_dict,
                                        'object': object_dict,
                                    }
                                    spo_list.append(spo)
            
                    batch_data[i]['entity_list'] = entity_list
                    batch_data[i]['spo_list'] = spo_list
                                          
            elif item['task_type'] in ['事件抽取'] and isinstance(item['choice'][0], dict):
                event_type_list, entity_type_list, args2event = self.data_encode.process_event_choice(item['choice'])
                
                event_args_type_list=[]
                for at,et_list in args2event.items():
                    for et in et_list:
                        if (et,at) not in event_args_type_list:
                            event_args_type_list.append((et,at))
                
                entity_logits = span_logits[i][:, :, :len(entity_type_list)]
                event_logits = span_logits[i][:, :, len(entity_type_list):len(entity_type_list)+len(event_type_list)]
                trigger_args_logits = span_logits[i][:,:,-1]

                entity_logits = np.tile(entity_logits[:, :, np.newaxis, :],[1,1,len(event_type_list),1])
                event_logits = np.tile(event_logits[:, :, :, np.newaxis],[1,1,1,len(entity_type_list)])
                event_entity_logits = (event_logits+entity_logits)/2
                seq_len,seq_len,etl,atl=event_entity_logits.shape
                for ei,et in enumerate(event_type_list):
                    for ai,at in enumerate(entity_type_list):
                        if (et,at) not in event_args_type_list:
                            event_entity_logits[:,:,ei,ai]=np.zeros((seq_len,seq_len))
                
                pred_event_type_list = []
                args_list=[]
                trigger_list=[]                
                for sidx in range(event_entity_logits.shape[0]):
                    for eidx in range(sidx, event_entity_logits.shape[1]):
                        event_entity_type_idx = np.unravel_index(np.argmax(event_entity_logits[sidx, eidx], axis=None), event_entity_logits[sidx, eidx].shape)
                        entity_type_score = entity_logits[sidx, eidx,event_entity_type_idx[0],event_entity_type_idx[1]]
                        event_type_score = event_logits[sidx, eidx,event_entity_type_idx[0],event_entity_type_idx[1]]
                        entity_type = entity_type_list[event_entity_type_idx[1]]
                        event_type = event_type_list[event_entity_type_idx[0]]
                        if entity_type_score+event_type_score>self.args.threshold_entity+self.args.threshold_event:
                            entity,start_idx,end_idx = self.extract_entity(item['text'],[sidx, eidx],query_len,offset_mapping)
                            if entity !='':
                                  
                                entity = {
                                    'entity_text': entity,
                                    'entity_type': entity_type,
                                    'entity_score':float(entity_type_score),
                                    'entity_index': [[start_idx, end_idx]]
                                }
                                event={}
                                event['event_type']=event_type                                
                                event['args']= [entity]
                                
                                if event_type not in pred_event_type_list:
                                    pred_event_type_list.append(event_type)
                        
                                if entity_type == '触发词':
                                    trigger_list.append((event ,sidx, eidx))
                                else:
                                    args_list.append((event, sidx, eidx))
                
                if len(trigger_list)+len(args_list)>sample_length/4:
                    batch_data[i]['event_list'] = []
                    continue            
                   
                event_list=[]
                for event_type in pred_event_type_list:
                    tmp_e_list=[]
                    trigger_idx_list=[]
                    for e in trigger_list:
                        if e[0]['event_type']==event_type:
                            tmp={}
                            tmp['event_type']=event_type
                            tmp['args']=e[0]['args']
                            tmp_e_list.append(tmp)
                            trigger_idx_list.append((e[1],e[2]))
                            
                    if tmp_e_list==[]:
                        tmp={}
                        tmp['event_type']=event_type
                        tmp['args']=[]
                        tmp_e_list.append(tmp)
                                              
                    for e in args_list:
                        if e[0]['event_type']==event_type:
                            if trigger_idx_list==[]:
                                tmp_e_list[0]['args'].extend(e[0]['args'])
                            else:
                                scores=[]
                                for t in trigger_idx_list:
                                    score=trigger_args_logits[e[1],t[0]]+trigger_args_logits[e[2],t[1]]
                                    scores.append(score)
                                et=scores.index(max(scores))
                                tmp_e_list[et]['args'].extend(e[0]['args'])
                    event_list.extend(tmp_e_list)
                batch_data[i]['event_list'] = event_list
        return batch_data


class ComputAcc:
    def __init__(self, args):
        self.args=args
        added_token = ['[unused'+str(i+1)+']' for i in range(99)]
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_path, is_split_into_words=True, add_prefix_space=True, additional_special_tokens=added_token)
        if args.fast_ex_mode:
            self.em = FastExtractModel(self.tokenizer, args)
        else:
            self.em = ExtractModel(self.tokenizer, args)
    
    def predict(self, test_data, model):
        test_data_ori=copy.deepcopy(test_data)
        result = []
        start = 0
        while start < len(test_data_ori):
            batch_data = test_data_ori[start:start+self.args.batchsize]
            start += self.args.batchsize
            batch_result = self.em.extract(
                batch_data, model)
            result.extend(batch_result)
        
        if isinstance(test_data[0]['choice'][0],list):
            f1, recall, precise = get_rel_f1(test_data, result)
        elif isinstance(test_data[0]['choice'][0],dict):
            f1, recall, precise = get_event_f1(test_data, result)
        else:
            f1, recall, precise = get_entity_f1(test_data, result)
            
        del test_data_ori, result
        gc.collect()
        
        return f1, recall, precise
 

class UniEXPipelines:
    @staticmethod
    def pipelines_args(parent_args):
        total_parser = parent_args.add_argument_group("piplines args")
        total_parser.add_argument(
            '--pretrained_model_path', default='', type=str)
        total_parser.add_argument('--output_path',
                                  default='./predict.json', type=str)

        total_parser.add_argument('--load_checkpoints_path',
                                  default='', type=str)

        total_parser.add_argument('--max_extract_entity_number',
                                  default=1, type=float)

        total_parser.add_argument('--train', action='store_true')
        total_parser.add_argument('--fast_ex_mode', action='store_true')

        
        
        total_parser.add_argument('--threshold_index',
                                  default=0.5, type=float)

        total_parser.add_argument('--threshold_entity',
                                  default=0.5, type=float)
        
        total_parser.add_argument('--threshold_event',
                                  default=0.5, type=float)

        total_parser.add_argument('--threshold_relation',
                                  default=0.5, type=float)

        total_parser = UniEXDataModel.add_data_specific_args(total_parser)
        total_parser = TaskModelCheckpoint.add_argparse_args(total_parser)
        total_parser = UniEXLitModel.add_model_specific_args(total_parser)
        total_parser = pl.Trainer.add_argparse_args(parent_args)
        return parent_args

    def __init__(self, args):

        if args.load_checkpoints_path != '':
            self.model = UniEXLitModel.load_from_checkpoint(
                args.load_checkpoints_path, args=args)
            print('导入模型成功：', args.load_checkpoints_path)
            
        else:
            self.model = UniEXLitModel(args)
            

        self.args = args
        self.checkpoint_callback = TaskModelCheckpoint(args).callbacks
        self.logger = loggers.TensorBoardLogger(save_dir=args.default_root_dir)
        self.trainer = pl.Trainer.from_argparse_args(args,
                                                     logger=self.logger,
                                                     callbacks=[self.checkpoint_callback])

        added_token = ['[unused'+str(i+1)+']' for i in range(10)]
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_path, is_split_into_words=True, add_prefix_space=True, additional_special_tokens=added_token)
        if args.fast_ex_mode:
            self.em = FastExtractModel(self.tokenizer, args)
        else:
            self.em = ExtractModel(self.tokenizer, args)

    def fit(self, train_data, dev_data,test_data=[]):
        data_model = UniEXDataModel(
            train_data, dev_data, self.tokenizer, self.args)
        self.model.num_data = len(train_data)
        self.model.dev_data = dev_data
        self.model.test_data = test_data
        self.trainer.fit(self.model, data_model)

    def predict(self, test_data, cuda=True):
        result = []
        start = 0
        if cuda:
            self.model = self.model.cuda()
        self.model.eval()
        while start < len(test_data):
            batch_data = test_data[start:start+self.args.batchsize]
            start += self.args.batchsize
            batch_result = self.em.extract(
                batch_data, self.model.model)
            result.extend(batch_result)
    
        return result


def load_data(data_path):
    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        samples = [json.loads(line) for line in tqdm(lines)]
    return samples


def main():
    total_parser = argparse.ArgumentParser("TASK NAME")
    total_parser.add_argument('--data_dir', default='./data', type=str)
    total_parser.add_argument('--train_data', default='train.json', type=str)
    total_parser.add_argument('--valid_data', default='dev.json', type=str)
    total_parser.add_argument('--test_data', default='test.json', type=str)
    total_parser = UniEXPipelines.pipelines_args(total_parser)
    args = total_parser.parse_args()

    train_data = load_data(os.path.join(args.data_dir, args.train_data))
    dev_data = load_data(os.path.join(args.data_dir, args.valid_data))
    test_data = load_data(os.path.join(args.data_dir, args.test_data))

    # train_data=train_data[:10]
    test_data=test_data[:100]
    dev_data=dev_data[:10]
    test_data_ori = copy.deepcopy(test_data)
    
    model = UniEXPipelines(args)
    if args.train:
        model.fit(train_data, dev_data,test_data)
    
    start_time=time.time()
    pred_data = model.predict(test_data)
    consum=time.time()-start_time
    print('总共耗费：',consum)
    print('sent/s：',len(test_data)/consum)

    for line in pred_data[:10]:
        print(line)

    if isinstance(test_data_ori[0]['choice'][0],list):
        f1, recall, precise = get_rel_f1(test_data_ori, pred_data)

        print('rel_f1:',f1)
        print('rel_recall:',recall)
        print('rel_precise:',precise)
        
    elif isinstance(test_data_ori[0]['choice'][0],dict):
        f1, recall, precise = get_event_f1(test_data_ori, pred_data)

        print('event_f1:',f1)
        print('event_recall:',recall)
        print('event_precise:',precise)
        
    f1, recall, precise = get_entity_f1(test_data_ori, pred_data)

    print('f1:',f1)
    print('recall:',recall)
    print('precise:',precise)



if __name__ == "__main__":
    main()
