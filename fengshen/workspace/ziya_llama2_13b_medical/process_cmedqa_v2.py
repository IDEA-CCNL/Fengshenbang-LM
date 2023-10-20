#coding=utf8
import json
question_path = '/cognitive_comp/ganruyi/datasets/medical_data/cMedQA2/question.csv'
answer_path = '/cognitive_comp/ganruyi/datasets/medical_data/cMedQA2/answer.csv'
test_path = '/cognitive_comp/ganruyi/datasets/medical_data/cMedQA2/test_candidates.txt'
cache_dir = '/cognitive_comp/ganruyi/datasets/'
from datasets import load_dataset
q_ds = load_dataset("csv", data_files=question_path, cache_dir=cache_dir)
a_ds = load_dataset("csv", data_files=answer_path, cache_dir=cache_dir)
test_ds = load_dataset("csv", data_files=test_path, cache_dir=cache_dir)
test_ds = test_ds['train'].filter(lambda example : example['label'] == 1)
print(q_ds)
print(a_ds)
print(test_ds)

q_dict = {}
for id, content in zip(q_ds['question_id'], q_ds['content']):
    q_dict[id] = content

print(len(q_dict))
'''
DatasetDict({
    train: Dataset({
        features: ['question_id', 'content'],
        num_rows: 120000
    })
})
DatasetDict({
    train: Dataset({
        features: ['ans_id', 'question_id', 'content'],
        num_rows: 226266
    })
})
Dataset({
    features: ['question_id', 'ans_id', 'cnt', 'label'],
    num_rows: 7552
})
'''


q_dict = {}
for id, content in zip(q_ds['train']['question_id'], q_ds['train']['content']):
    q_dict[id] = content

print(len(q_dict))
a_dict = {}
for id, content in zip(a_ds['train']['ans_id'], a_ds['train']['content']):
    a_dict[id] = content

print(len(a_dict))

output_path = '/cognitive_comp/ganruyi/datasets/medical_data/cMedQA2/test_processed.json'
out_file = open(output_path, 'w')
for aid, qid in zip(test_ds['ans_id'], test_ds['question_id']):
    print(aid, qid)
    out_data = {}
    out_data['prompt'] = [q_dict[qid]]
    out_data['output'] = [a_dict[aid]]
    out_file.write(json.dumps(out_data, ensure_ascii=False))
    out_file.write('\n')