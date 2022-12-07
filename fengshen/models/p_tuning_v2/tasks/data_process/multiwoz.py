# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MultiWOZ 2.1 for DST, seq2seq methods specially."""

import difflib
import re
import shutil
from shutil import copyfile

import datasets
import numpy as np

# we adopt the seq2seq handling method from Zero-shot T5, we adopted the slot-type as prompt words
# , for more information, see:
# paper: https://www.aclweb.org/anthology/2021.naacl-main.448.pdf
# code: https://github.com/facebookresearch/Zero-Shot-DST/tree/main/T5DST

np.set_printoptions(precision=3)

np.random.seed(2)

'''
Most of the codes are from https://github.com/budzianowski/multiwoz
'''

# GLOBAL VARIABLES
DICT_SIZE = 400
MAX_LENGTH = 50
IGNORE_KEYS_IN_GOAL = ['eod', 'topic', 'messageLen', 'message']

fin = open('third_party/zero_shot_dst/T5DST/utils/mapping.pair', 'r')
replacements = []
for line in fin.readlines():
    tok_from, tok_to = line.replace('\n', '').split('\t')
    replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text


def normalize(text, clean_value=True):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    if clean_value:
        # normalize phone number
        ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m[0], sidx)
                if text[sidx - 1] == '(':
                    sidx -= 1
                eidx = text.find(m[-1], sidx) + len(m[-1])
                text = text.replace(text[sidx:eidx], ''.join(m))

        # normalize postcode
        ms = re.findall(
            '([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
            text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m, sidx)
                eidx = sidx + len(m)
                text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    if clean_value:
        # replace time and and price
        text = re.sub(timepat, ' [value_time] ', text)
        text = re.sub(pricepat, ' [value_price] ', text)
        # text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\"\<>@\(\)]', '', text)  # remove

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text


def fixDelex(filename, data, data2, idx, idx_acts):
    """Given system dialogue acts fix automatic delexicalization."""
    try:
        turn = data2[filename.strip('.json')][str(idx_acts)]
    except:
        return data

    if not isinstance(turn, str):  # and not isinstance(turn, unicode):
        for k, act in turn.items():
            if 'Attraction' in k:
                if 'restaurant_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("restaurant", "attraction")
                if 'hotel_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("hotel", "attraction")
            if 'Hotel' in k:
                if 'attraction_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("attraction", "hotel")
                if 'restaurant_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("restaurant", "hotel")
            if 'Restaurant' in k:
                if 'attraction_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("attraction", "restaurant")
                if 'hotel_' in data['log'][idx]['text']:
                    data['log'][idx]['text'] = data['log'][idx]['text'].replace("hotel", "restaurant")

    return data


def getDialogueAct(filename, data, data2, idx, idx_acts):
    """Given system dialogue acts fix automatic delexicalization."""
    acts = []
    try:
        turn = data2[filename.strip('.json')][str(idx_acts)]
    except:
        return acts

    if not isinstance(turn, str):  # and not isinstance(turn, unicode):
        for k in turn.keys():
            # temp = [k.split('-')[0].lower(), k.split('-')[1].lower()]
            # for a in turn[k]:
            #     acts.append(temp + [a[0].lower()])

            if k.split('-')[1].lower() == 'request':
                for a in turn[k]:
                    acts.append(a[0].lower())
            elif k.split('-')[1].lower() == 'inform':
                for a in turn[k]:
                    acts.append([a[0].lower(), normalize(a[1].lower(), False)])

    return acts


def get_summary_bstate(bstate, get_domain=False):
    """Based on the mturk annotations we form multi-domain belief state"""
    domains = [u'taxi', u'restaurant', u'hospital', u'hotel', u'attraction', u'train', u'police']
    summary_bstate = []
    summary_bvalue = []
    active_domain = []
    for domain in domains:
        domain_active = False

        booking = []
        # print(domain,len(bstate[domain]['book'].keys()))
        for slot in sorted(bstate[domain]['book'].keys()):
            if slot == 'booked':
                if len(bstate[domain]['book']['booked']) != 0:
                    booking.append(1)
                    # summary_bvalue.append("book {} {}:{}".format(domain, slot, "Yes"))
                else:
                    booking.append(0)
            else:
                if bstate[domain]['book'][slot] != "":
                    booking.append(1)
                    summary_bvalue.append(["{}-book {}".format(domain, slot.strip().lower()),
                                           normalize(bstate[domain]['book'][slot].strip().lower(),
                                                     False)])  # (["book", domain, slot, bstate[domain]['book'][slot]])
                else:
                    booking.append(0)
        if domain == 'train':
            if 'people' not in bstate[domain]['book'].keys():
                booking.append(0)
            if 'ticket' not in bstate[domain]['book'].keys():
                booking.append(0)
        summary_bstate += booking

        for slot in bstate[domain]['semi']:
            slot_enc = [0, 0, 0]  # not mentioned, dontcare, filled
            if bstate[domain]['semi'][slot] == 'not mentioned':
                slot_enc[0] = 1
            elif bstate[domain]['semi'][slot] in ['dont care', 'dontcare', "don't care", "do not care"]:
                slot_enc[1] = 1
                summary_bvalue.append(
                    ["{}-{}".format(domain, slot.strip().lower()), "dontcare"])  # (["semi", domain, slot, "dontcare"])
            elif bstate[domain]['semi'][slot]:
                summary_bvalue.append(["{}-{}".format(domain, slot.strip().lower()),
                                       normalize(bstate[domain]['semi'][slot].strip().lower(),
                                                 False)])  # (["semi", domain, slot, bstate[domain]['semi'][slot]])
            if slot_enc != [0, 0, 0]:
                domain_active = True
            summary_bstate += slot_enc

        # quasi domain-tracker
        if domain_active:
            summary_bstate += [1]
            active_domain.append(domain)
        else:
            summary_bstate += [0]

    # print(len(summary_bstate))
    assert len(summary_bstate) == 94
    if get_domain:
        return active_domain
    else:
        return summary_bstate, summary_bvalue


def analyze_dialogue(dialogue, maxlen):
    """Cleaning procedure for all kinds of errors in text and annotation."""
    d = dialogue
    # do all the necessary postprocessing
    if len(d['log']) % 2 != 0:
        # print path
        print('odd # of turns')
        return None  # odd number of turns, wrong dialogue
    d_pp = {}
    d_pp['goal'] = d['goal']  # for now we just copy the goal
    usr_turns = []
    sys_turns = []
    # last_bvs = []
    for i in range(len(d['log'])):
        if len(d['log'][i]['text'].split()) > maxlen:
            # print('too long')
            return None  # too long sentence, wrong dialogue
        if i % 2 == 0:  # usr turn
            text = d['log'][i]['text']
            if not is_ascii(text):
                # print('not ascii')
                return None
            usr_turns.append(d['log'][i])
        else:  # sys turn
            text = d['log'][i]['text']
            if not is_ascii(text):
                # print('not ascii')
                return None
            belief_summary, belief_value_summary = get_summary_bstate(d['log'][i]['metadata'])
            d['log'][i]['belief_summary'] = str(belief_summary)
            d['log'][i]['belief_value_summary'] = belief_value_summary
            sys_turns.append(d['log'][i])
    d_pp['usr_log'] = usr_turns
    d_pp['sys_log'] = sys_turns

    return d_pp


def get_dial(dialogue):
    """Extract a dialogue from the file"""
    dial = []
    d_orig = analyze_dialogue(dialogue, MAX_LENGTH)  # max turn len is 50 words
    if d_orig is None:
        return None
    usr = [t['text'] for t in d_orig['usr_log']]
    sys = [t['text'] for t in d_orig['sys_log']]
    sys_a = [t['dialogue_acts'] for t in d_orig['sys_log']]
    bvs = [t['belief_value_summary'] for t in d_orig['sys_log']]
    domain = [t['domain'] for t in d_orig['usr_log']]
    for item in zip(usr, sys, sys_a, domain, bvs):
        dial.append({'usr': item[0], 'sys': item[1], 'sys_a': item[2], 'domain': item[3], 'bvs': item[4]})
    return dial


def loadData(root_path):
    dir_name = 'MULTIWOZ2.1'
    shutil.copy(os.path.join(root_path, dir_name, 'data.json'), root_path)
    shutil.copy(os.path.join(root_path, dir_name, 'ontology.json'), root_path)
    shutil.copy(os.path.join(root_path, dir_name, 'valListFile.json'), root_path)
    shutil.copy(os.path.join(root_path, dir_name, 'testListFile.json'), root_path)
    shutil.copy(os.path.join(root_path, dir_name, 'dialogue_acts.json'), root_path)


def getDomain(idx, log, domains, last_domain):
    if idx == 1:
        active_domains = get_summary_bstate(log[idx]["metadata"], True)
        crnt_doms = active_domains[0] if len(active_domains) != 0 else domains[0]
        return crnt_doms
    else:
        ds_diff = get_ds_diff(log[idx - 2]["metadata"], log[idx]["metadata"])
        if len(ds_diff.keys()) == 0:  # no clues from dialog states
            crnt_doms = last_domain
        else:
            crnt_doms = list(ds_diff.keys())
        # print(crnt_doms)
        return crnt_doms[0]  # How about multiple domains in one sentence senario ?


def get_ds_diff(prev_d, crnt_d):
    diff = {}
    # Sometimes, metadata is an empty dictionary, bug?
    if not prev_d or not crnt_d:
        return diff

    for ((k1, v1), (k2, v2)) in zip(prev_d.items(), crnt_d.items()):
        assert k1 == k2
        if v1 != v2:  # updated
            diff[k2] = v2
    return diff


def createData(root_path):
    # download the data
    loadData(root_path)

    # create dictionary of delexicalied values that then we will search against, order matters here!
    # dic = delexicalize.prepareSlotValuesIndependent()
    delex_data = {}

    fin1 = open(os.path.join(root_path, 'data.json'), 'r')
    data = json.load(fin1)

    fin2 = open(os.path.join(root_path, 'dialogue_acts.json'), 'r')
    data2 = json.load(fin2)

    for didx, dialogue_name in enumerate(data):

        dialogue = data[dialogue_name]

        domains = []
        for dom_k, dom_v in dialogue['goal'].items():
            if dom_v and dom_k not in IGNORE_KEYS_IN_GOAL:  # check whether contains some goal entities
                domains.append(dom_k)

        idx_acts = 1
        last_domain, last_slot_fill = "", []
        for idx, turn in enumerate(dialogue['log']):
            # normalization, split and delexicalization of the sentence
            origin_text = normalize(turn['text'], False)
            # origin_text = delexicalize.markEntity(origin_text, dic)
            dialogue['log'][idx]['text'] = origin_text

            if idx % 2 == 1:  # if it's a system turn

                cur_domain = getDomain(idx, dialogue['log'], domains, last_domain)
                last_domain = [cur_domain]

                dialogue['log'][idx - 1]['domain'] = cur_domain
                dialogue['log'][idx]['dialogue_acts'] = getDialogueAct(dialogue_name, dialogue, data2, idx, idx_acts)
                idx_acts += 1

            # FIXING delexicalization:
            dialogue = fixDelex(dialogue_name, dialogue, data2, idx, idx_acts)

        delex_data[dialogue_name] = dialogue

        # if didx > 10:
        #     break

    # with open('data/multi-woz/woz2like_data.json', 'w') as outfile:
    #     json.dump(delex_data, outfile)

    return delex_data


def buildDelexDict(origin_sent, delex_sent):
    dictionary = {}
    s = difflib.SequenceMatcher(None, delex_sent.split(), origin_sent.split())
    bs = s.get_matching_blocks()
    for i, b in enumerate(bs):
        if i < len(bs) - 2:
            a_start = b.a + b.size
            b_start = b.b + b.size
            b_end = bs[i + 1].b
            dictionary[a_start] = " ".join(origin_sent.split()[b_start:b_end])
    return dictionary


def divideData(data, root_path, target_path):
    """Given test and validation sets, divide
    the data for three different sets"""
    root_path = root_path

    os.makedirs(target_path, exist_ok=True)

    copyfile(os.path.join(root_path, 'ontology.json'), os.path.join(target_path, 'ontology.json'))

    testListFile = []
    fin = open(os.path.join(root_path, 'testListFile.json'), 'r')
    for line in fin:
        testListFile.append(line[:-1])
    fin.close()

    valListFile = []
    fin = open(os.path.join(root_path, 'valListFile.json'), 'r')
    for line in fin:
        valListFile.append(line[:-1])
    fin.close()

    trainListFile = open(os.path.join(target_path, 'trainListFile'), 'w')

    test_dials = []
    val_dials = []
    train_dials = []

    # dictionaries
    word_freqs_usr = OrderedDict()
    word_freqs_sys = OrderedDict()

    count_train, count_val, count_test = 0, 0, 0

    ontology = {}

    for dialogue_name in data:
        # print dialogue_name
        dial_item = data[dialogue_name]
        domains = []
        for dom_k, dom_v in dial_item['goal'].items():
            if dom_v and dom_k not in IGNORE_KEYS_IN_GOAL:  # check whether contains some goal entities
                domains.append(dom_k)

        turn_exmaple = {"system": "none", "user": "none", "state": {"active_intent": "none", "slot_values": {}}}
        dial = get_dial(data[dialogue_name])
        if dial:
            dial_example = {"dial_id": dialogue_name, "domains": list(set(domains)), "turns": []}
            # dialogue = {}
            # dialogue['dialogue_idx'] = dialogue_name
            # dialogue['domains'] = list(set(domains)) #list(set([d['domain'] for d in dial]))
            # last_bs = []
            # dialogue['dialogue'] = []

            for turn_i, turn in enumerate(dial):
                # usr, usr_o, sys, sys_o, sys_a, domain
                turn_exmaple = {"system": "none", "user": "none", "state": {"active_intent": "none", "slot_values": {}}}
                turn_exmaple['system'] = dial[turn_i - 1]['sys'] if turn_i > 0 else "none"
                turn_exmaple['state']["slot_values"] = {s[0]: s[1] for s in turn['bvs']}
                turn_exmaple['user'] = turn['usr']
                dial_example['turns'].append(turn_exmaple)

                for ss, vv in turn_exmaple['state']["slot_values"].items():
                    if ss not in ontology:
                        ontology[ss] = []
                    if vv not in ontology[ss]:
                        ontology[ss].append(vv)

            if dialogue_name in testListFile:
                test_dials.append(dial_example)
                count_test += 1
            elif dialogue_name in valListFile:
                val_dials.append(dial_example)
                count_val += 1
            else:
                trainListFile.write(dialogue_name + '\n')
                train_dials.append(dial_example)
                count_train += 1

    print("# of dialogues: Train {}, Val {}, Test {}".format(count_train, count_val, count_test))

    # save all dialogues
    with open(os.path.join(target_path, "train_dials.json"), 'w') as f:
        json.dump(train_dials, f, indent=4)

    with open(os.path.join(target_path, "dev_dials.json"), 'w') as f:
        json.dump(val_dials, f, indent=4)

    with open(os.path.join(target_path, "test_dials.json"), 'w') as f:
        json.dump(test_dials, f, indent=4)

    with open(os.path.join(target_path, 'ontology.json'), 'w') as f:
        json.dump(ontology, f, indent=4)


# from data_loader import *


import json
from torch.utils.data import DataLoader, Dataset
import os
import random
from functools import partial
from third_party.zero_shot_dst.T5DST.utils.fix_label import fix_general_label_error
from collections import OrderedDict

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]

random.seed(577)
HISTORY_MAX_LEN = 450
GPT_MAX_LEN = 1024


class DSTDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, args):
        """Reads source and target sequences from txt files."""
        self.data = data
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item_info = self.data[index]
        if self.args["slot_lang"] == "value":
            random.shuffle(item_info["value_list"])
            item_info["intput_text"] += " is " + " or ".join(item_info["value_list"]) + " or none?"
        return item_info

    def __len__(self):
        return len(self.data)


def read_data(args, path_name, SLOTS, tokenizer, description, dataset=None):
    slot_lang_list = ["description_human", "rule_description", "value_description", "rule2", "rule3"]
    print(("Reading all files from {}".format(path_name)))
    data = []
    domain_counter = {}
    # read files
    with open(path_name) as f:
        dials = json.load(f)

        if dataset == "train" and args["fewshot"] > 0:
            random.Random(args["seed"]).shuffle(dials)
            dials = dials[:int(len(dials) * args["fewshot"])]

        for dial_dict in dials:
            dialog_history = ""

            # Counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:
                    continue
                if domain not in domain_counter.keys():
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if args["only_domain"] != "none" and args["only_domain"] not in dial_dict["domains"]:
                continue
            if (args["except_domain"] != "none" and dataset == "test" and args["except_domain"] not in dial_dict[
                "domains"]) or \
                    (args["except_domain"] != "none" and dataset != "test" and [args["except_domain"]] == dial_dict[
                        "domains"]):
                continue

            # Reading data
            for ti, turn in enumerate(dial_dict["turns"]):
                turn_id = ti

                # accumulate dialogue utterances
                dialog_history += (" System: " + turn["system"] + " User: " + turn["user"])
                if args["fix_label"]:
                    slot_values = fix_general_label_error(turn["state"]["slot_values"], SLOTS)
                else:
                    slot_values = turn["state"]["slot_values"]
                # input: dialogue history + slot
                # output: value

                # Generate domain-dependent slot list
                slot_temp = SLOTS
                if dataset == "train" or dataset == "dev":
                    if args["except_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["except_domain"] not in k]
                        slot_values = OrderedDict(
                            [(k, v) for k, v in slot_values.items() if args["except_domain"] not in k])
                    elif args["only_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["only_domain"] in k])
                else:
                    if args["except_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["except_domain"] in k]
                        slot_values = OrderedDict(
                            [(k, v) for k, v in slot_values.items() if args["except_domain"] in k])
                    elif args["only_domain"] != "none":
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        slot_values = OrderedDict([(k, v) for k, v in slot_values.items() if args["only_domain"] in k])

                turn_belief_list = [str(k) + '-' + str(v) for k, v in slot_values.items()]

                # baseline gpt have different preprocessing, e.g., output: (slot1-value1, slot2-value2, slot3-value3, ...)
                if "gpt" in args["model_name"]:
                    turn_slots = []
                    turn_slot_values = []
                    if len(dialog_history.split()) > 800:
                        continue
                    for slot in slot_temp:
                        # skip unrelevant slots for out of domain setting
                        if args["except_domain"] != "none" and dataset != "test":
                            if slot.split("-")[0] not in dial_dict["domains"]:
                                continue
                        input_text = dialog_history + f" {tokenizer.sep_token} {slot}" + " " + tokenizer.bos_token
                        output_text = input_text + " " + turn["state"]["slot_values"].get(slot,
                                                                                          'none').strip() + " " + tokenizer.eos_token
                        slot_text = slot
                        value_text = turn["state"]["slot_values"].get(slot, 'none').strip()

                        data_detail = {
                            "ID": dial_dict["dial_id"],
                            "domains": dial_dict["domains"],
                            "turn_id": turn_id,
                            "dialog_history": dialog_history,
                            "turn_belief": turn_belief_list,
                            "intput_text": input_text,
                            "output_text": output_text,
                            "slot_text": slot_text,
                            "value_text": value_text
                        }
                        data.append(data_detail)

                else:
                    for slot in slot_temp:

                        # skip unrelevant slots for out of domain setting
                        if args["except_domain"] != "none" and dataset != "test":
                            if slot.split("-")[0] not in dial_dict["domains"]:
                                continue

                        output_text = slot_values.get(slot, 'none').strip() + f" {tokenizer.eos_token}"
                        slot_text = slot
                        value_text = slot_values.get(slot, 'none').strip()

                        if args["slot_lang"] == "human":
                            slot_lang = description[slot]["description_human"]
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                        elif args["slot_lang"] == "naive":
                            slot_lang = description[slot]["naive"]
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                        elif args["slot_lang"] == "value":
                            slot_lang = description[slot]["naive"]
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}"
                        elif args["slot_lang"] == "question":
                            slot_lang = description[slot]["question"]
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}"
                        elif args["slot_lang"] == "slottype":
                            slot_lang = description[slot]["slottype"]
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot_lang}?"
                        else:
                            input_text = dialog_history + f" {tokenizer.sep_token} {slot}"

                        data_detail = {
                            "ID": dial_dict["dial_id"],
                            "domains": dial_dict["domains"],
                            "turn_id": turn_id,
                            "dialog_history": dialog_history,
                            "turn_belief": turn_belief_list,
                            "intput_text": input_text,
                            "output_text": output_text,
                            "slot_text": slot_text,
                            "value_text": value_text,
                            "value_list": description[slot]["values"]
                        }
                        data.append(data_detail)
    # print(len(data))
    for idx in range(10):
        print(data[idx])
    print("domain_counter", domain_counter)
    return data, slot_temp


def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]

    return SLOTS


def gpt_collate_fn(data, tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    output_batch = tokenizer(batch_data["output_text"], padding=True, return_tensors="pt", add_special_tokens=False,
                             return_attention_mask=False, truncation=True, max_length=1000)
    batch_data["input_ids"] = output_batch['input_ids']
    return batch_data


def collate_fn(data, tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_batch = tokenizer(batch_data["intput_text"], padding=True, return_tensors="pt", add_special_tokens=False,
                            verbose=False)
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    output_batch = tokenizer(batch_data["output_text"], padding=True, return_tensors="pt", add_special_tokens=False,
                             return_attention_mask=False)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids'] == tokenizer.pad_token_id, -100)
    batch_data["decoder_output"] = output_batch['input_ids']

    return batch_data


def prepare_data(args, tokenizer):
    path_train = 'data/train_dials.json'
    path_dev = 'data/dev_dials.json'
    path_test = 'data/test_dials.json'

    ontology = json.load(open("data/multi-woz/MULTIWOZ2 2/ontology.json", 'r'))
    ALL_SLOTS = get_slot_information(ontology)
    description = json.load(open(SLOT_DESCRIPTION_PATH, 'r'))

    data_train, _ = read_data(args, path_train, ALL_SLOTS, tokenizer, description, "train")
    data_dev, _ = read_data(args, path_dev, ALL_SLOTS, tokenizer, description, "dev")
    data_test, ALL_SLOTS = read_data(args, path_test, ALL_SLOTS, tokenizer, description, "test")

    train_dataset = DSTDataset(data_train, args)
    dev_dataset = DSTDataset(data_dev, args)
    test_dataset = DSTDataset(data_test, args)

    if "gpt" in args["model_name"]:
        train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True,
                                  collate_fn=partial(gpt_collate_fn, tokenizer=tokenizer), num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False,
                                 collate_fn=partial(gpt_collate_fn, tokenizer=tokenizer), num_workers=16)
        dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False,
                                collate_fn=partial(gpt_collate_fn, tokenizer=tokenizer), num_workers=16)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args["train_batch_size"], shuffle=True,
                                  collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
        test_loader = DataLoader(test_dataset, batch_size=args["test_batch_size"], shuffle=False,
                                 collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
        dev_loader = DataLoader(dev_dataset, batch_size=args["dev_batch_size"], shuffle=False,
                                collate_fn=partial(collate_fn, tokenizer=tokenizer), num_workers=16)
    fewshot_loader_dev = None
    fewshot_loader_test = None
    return train_loader, dev_loader, test_loader, ALL_SLOTS, fewshot_loader_dev, fewshot_loader_test


SLOT_DESCRIPTION_PATH = "third_party/zero_shot_dst/T5DST/utils/slot_description.json"

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """
[Eric et al. 2019]
@article{eric2019multiwoz,
  title={MultiWOZ 2.1: Multi-Domain Dialogue State Corrections and State Tracking Baselines},
  author={Eric, Mihail and Goel, Rahul and Paul, Shachi and Sethi, Abhishek and Agarwal, Sanchit and Gao, Shuyag and Hakkani-Tur, Dilek},
  journal={arXiv preprint arXiv:1907.01669},
  year={2019}
}
@InProceedings{WuTradeDST2019,
  	author = "Wu, Chien-Sheng and Madotto, Andrea and Hosseini-Asl, Ehsan and Xiong, Caiming and Socher, Richard and Fung, Pascale",
  	title = 	"Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems",
  	booktitle = 	"Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  	year = 	"2019",
  	publisher = "Association for Computational Linguistics"
}
@inproceedings{lin2021leveraging,
  title={Leveraging Slot Descriptions for Zero-Shot Cross-Domain Dialogue StateTracking},
  author={Lin, Zhaojiang and Liu, Bing and Moon, Seungwhan and Crook, Paul A and Zhou, Zhenpeng and Wang, Zhiguang and Yu, Zhou and Madotto, Andrea and Cho, Eunjoon and Subba, Rajen},
  booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={5640--5648},
  year={2021}
}

"""

_DESCRIPTION = """
There are 3,406 single-domain dialogues that include booking if the domain allows for that and 7,032 multi-domain dialogues consisting of at least 2 up to 5 domains. To enforce reproducibility of results, the corpus was randomly split into a train, test and development set. The test and development sets contain 1k examples each. Even though all dialogues are coherent, some of them were not finished in terms of task description. Therefore, the validation and test sets only contain fully successful dialogues thus enabling a fair comparison of models. There are no dialogues from hospital and police domains in validation and testing sets.

Each dialogue consists of a goal, multiple user and system utterances as well as a belief state. Additionally, the task description in natural language presented to turkers working from the visitor’s side is added. Dialogues with MUL in the name refers to multi-domain dialogues. Dialogues with SNG refers to single-domain dialogues (but a booking sub-domain is possible). The booking might not have been possible to complete if fail_book option is not empty in goal specifications – turkers did not know about that.

The belief state have three sections: semi, book and booked. Semi refers to slots from a particular domain. Book refers to booking slots for a particular domain and booked is a sub-list of book dictionary with information about the booked entity (once the booking has been made). The goal sometimes was wrongly followed by the turkers which may results in the wrong belief state. The joint accuracy metrics includes ALL slots.
"""

_LICENSE = "Apache License 2.0"

URL = "https://www.repository.cam.ac.uk/bitstream/handle/1810/294507/MULTIWOZ2.1.zip?sequence=1&isAllowed=y"


class MultiWozV2_1(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("2.0.0")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fix_label = True  # We default it to fix the label

    def _info(self):
        features = datasets.Features(
            {
                "ID": datasets.Value("string"),
                "turn_id": datasets.Value("int32"),
                "ontology_path": datasets.Value("string"),
                "dialog": {
                    "sys": datasets.Sequence(datasets.Value("string")),
                    "usr": datasets.Sequence(datasets.Value("string"))
                },
                "domains": datasets.Sequence(datasets.Value("string")),
                "ontology_slots": datasets.Sequence(datasets.Value("string")),
                "ontology_values": datasets.Sequence(datasets.Sequence(datasets.Value("string"))),
                "turn_belief": datasets.Sequence(datasets.Value("string")),
                "expanded_turn_belief": datasets.Sequence(
                    {
                        "slot": datasets.Value("string"),
                        "value": datasets.Value("string")
                    })
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,  # Here we define them above because they are different between the two configurations
            supervised_keys=None,
            homepage="https://github.com/budzianowski/multiwoz",
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # the code is from https://github.com/facebookresearch/Zero-Shot-DST/blob/main/T5DST/create_data.py
        print('Create WOZ-like dialogues. Get yourself a coffee, this might take a while.')
        # These words is for saluting MultiWOZ, so many error has been made in a time of coffee.
        root_path = dl_manager.download_and_extract(URL)
        target_path = os.path.join(root_path, "processed_data")
        if not os.path.exists(target_path):
            delex_data = createData(root_path)
            print('Divide dialogues...')
            divideData(delex_data, root_path, target_path)

        return [
            datasets.SplitGenerator(
                name=split_enum,
                gen_kwargs={
                    "root_path": target_path,  # the processed woz data as the new root data path
                    "file_path": file_path,
                    "section": section
                },
            )
            for file_path, split_enum, section in [
                (os.path.join(target_path, 'train_dials.json'), datasets.Split.TRAIN, "train"),
                (os.path.join(target_path, 'dev_dials.json'), datasets.Split.VALIDATION, "dev"),
                (os.path.join(target_path, 'test_dials.json'), datasets.Split.TEST, "test"),
            ]
        ]

    def _generate_examples(self, root_path, file_path, section):
        datasets_idx = -1
        ontology_path = os.path.join(root_path, "ontology.json")
        SLOTS = get_slot_information(json.load(open(ontology_path, 'r')))

        ontology_slots = []
        ontology_values = []
        description = json.load(open(SLOT_DESCRIPTION_PATH, 'r'))
        for slot in SLOTS:
            ontology_slots.append(slot)
            ontology_values.append(description[slot]["values"])

        domain_counter = {}

        # read files
        with open(file_path) as f:
            dials = json.load(f)
            for dial_dict in dials:
                dialog = {"sys": [], "usr": []}

                # Counting domains
                for domain in dial_dict["domains"]:
                    if domain not in EXPERIMENT_DOMAINS:
                        continue
                    if domain not in domain_counter.keys():
                        domain_counter[domain] = 0
                    domain_counter[domain] += 1

                # Reading data

                for ti, turn in enumerate(dial_dict["turns"]):
                    turn_id = ti

                    # accumulate dialogue utterances
                    dialog["sys"].append(turn["system"])
                    dialog["usr"].append(turn["user"])

                    # slot_values is the dst representation of this turn
                    if self.fix_label:
                        slot_values = fix_general_label_error(turn["state"]["slot_values"], SLOTS)
                    else:
                        slot_values = turn["state"]["slot_values"]

                    # Generate domain-dependent slot list
                    turn_belief_list = [str(k) + '-' + str(v) for k, v in slot_values.items()]

                    # SimpleTOD team(https://github.com/salesforce/simpletod) find that
                    # "For this task, we include none slot values in the sequence. We observed that this will improve SimpleTOD performance on DST by reducing false positive rates."
                    expanded_turn_belief_list = []
                    for slot in SLOTS:
                        expanded_turn_belief_list.append({"slot": str(slot), "value": str(slot_values.get(slot, 'none').strip())})

                    data_detail = {
                        "ID": dial_dict["dial_id"],
                        "turn_id": turn_id,
                        "ontology_path": ontology_path,
                        "domains": dial_dict["domains"],
                        "dialog": dialog,
                        "ontology_slots": ontology_slots,
                        "ontology_values": ontology_values,
                        "turn_belief": turn_belief_list,
                        "expanded_turn_belief": expanded_turn_belief_list
                    }
                    datasets_idx += 1
                    yield datasets_idx, data_detail
