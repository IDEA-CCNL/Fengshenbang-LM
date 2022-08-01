import json
import pandas as pd
import random
from pathlib import Path

def text_filtering(text):
    startIndex = text.find('“')
    if startIndex == -1:
        endIndex = text.find('”')
    while startIndex != -1: #i.e. if the first quote was found
        endIndex = text.find('”', startIndex + 1)
        if endIndex != -1:
            startIndex = text.find('“', endIndex + 1)
            if startIndex == -1:
                endIndex = text.find('”', endIndex + 1)
        else:
            break
    output_text = text[:endIndex] if endIndex != -1 else text
    # output_text = ''.join(output_text.split())
    return output_text

def get_raw_data(raw_data):
    train_data = {}
    with open(raw_data, 'r', encoding='utf8') as fh:
        for line in fh:
            line = json.loads(line)
            for key in line.keys():
                if key not in train_data.keys():
                    train_data[key] = [line[key]]
                else:
                    train_data[key].append(line[key])
    return train_data


def merge_sim_data(file_path, output_file):
    text_gen = []
    for file in file_path:
        with open(file, 'r', encoding='utf8') as fh:
            for line in fh:
                try:
                    text = json.loads(line)
                except:
                    pass
                textb = text['text_b']
                textb = (textb.split('” ')[0]).split('<|endoftext|>')[0]
                text_gen.append({'text_a':text['text_a'], 'text_b':textb, 'score':float(text['score']) if 'score' in text.keys() else 0.5})
    random.shuffle(text_gen)

    with open(output_file, 'w', encoding='utf-8') as f:
        for text in text_gen:
            json.dump(text, f, ensure_ascii=False)
            f.write('\n')


def save_gen_data(output_file, gened_dataset, shuffle=True):
    listy = gened_dataset.columns
    if shuffle:
        gened_dataset = gened_dataset.sample(frac=1).reset_index(drop=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for row in gened_dataset.itertuples():
            otc = {}
            i = 1
            for key in listy:
                otc[key] = row[i]
                i += 1
            json.dump(otc, f, ensure_ascii=False)
            f.write('\n')

# from thefuzz import process

def merge_gen_data(raw_data, sim_data, output_file, key='sentence', topk=1000, has_score=True):
    train_data = get_raw_data(raw_data)
    #     train_data[key] = [text[:10] for text in train_data[key]]

    res_keys = list(train_data.keys())
    train_data = pd.DataFrame(train_data)

    text_gen = {key:[], "text_b":[], "score":[]}
#     file = 'csldcp_few_total_dataset_filtered_0.9.jsonl'

    with open(sim_data, 'r', encoding='utf8') as fh:
        for line in fh:
            text = json.loads(line)
            if (len(text["text_b"]) < len(text["text_a"])/2 and len(text["text_b"]) < 20):
                continue
            if (text["text_a"] in text["text_b"]) or (text["text_b"] in text["text_a"]):
                continue
            # if float(text["score"]) < 0.5 or float(text["score"]) > 0.9:
            if has_score and float(text["score"]) < 0.2:
                continue
            text_gen[key].append(text["text_a"])
#             text_gen[key].append(text["text_a"][1:11])

            text_gen["text_b"].append(text["text_b"])
            text_gen["score"].append(float(text["score"] if has_score else 0.8))
    gen_data_filtered = pd.DataFrame(text_gen)
    gen_data_filtered = gen_data_filtered.sort_values([key, 'score'], ascending=False).groupby(key).head(topk)
    res_keys.remove(key)
    # fuzzy match key
#     best_key = lambda x: process.extractOne(x, train_data[key])[2]
#     print(f'keys {gen_data_filtered[key].map(best_key).values}')
#     gen_data_filtered[res_keys] = train_data.loc[gen_data_filtered[key].map(best_key).values, res_keys].values
#     gen_data_filtered = gen_data_filtered[res_keys].dropna().drop_duplicates(subset=['text_b'])
    
    res_keys.append('text_b')
    gen_data_filtered = gen_data_filtered.merge(train_data, on=key, how='left')[res_keys].dropna().drop_duplicates(subset=['text_b'])
    gen_data_filtered = gen_data_filtered.rename(columns={'text_b':key})

    save_gen_data(output_file, gen_data_filtered)


if __name__ == '__main__':
    raw_data = '/raid/wanghao/datasets/csldcp/train_0.json'
    sim_data = '/raid/wanghao/workspace/invTXL2/out/csldcp_few_total_dataset_filtered_0.9.jsonl'
    output_file = '/raid/wanghao/workspace/invTXL2/out/inv_csldcp_few_dataset_filtered_0.9.jsonl'
    merge_gen_data(raw_data, sim_data, output_file, key='content')
