#coding=utf8
import json

tasks = ['clinical_medicine','basic_medicine']
tasks_zh = ['临床医学', '基础医学']
dev_path = '/cognitive_comp/yangping/data/eval_data/ceval/ceval_json/dev.json'
val_path = '/cognitive_comp/yangping/data/eval_data/ceval/ceval_json/val.json'
out_val_path = '/cognitive_comp/ganruyi/datasets/medical_data/ceval_medical/val.json'
out_dev_path = '/cognitive_comp/ganruyi/datasets/medical_data/ceval_medical/dev.json'
def process(input_file, out_file):
    out = open(out_file, 'w')
    with open(input_file, 'r') as file:
        for line in file.readlines():
            data = json.loads(line)
            type = data['type']
            if type in tasks_zh:
                out.write(line)
process(val_path, out_val_path)
process(dev_path, out_dev_path)