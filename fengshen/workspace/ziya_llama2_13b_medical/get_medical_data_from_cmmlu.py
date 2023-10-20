#coding=utf8
import json

tasks = ['anatomy', 'clinical_knowledge', 'college_medical_statistics', 'genetics', 'nutrition', 'traditional_chinese_medicine', 'virology']
tasks_zh = ['解剖学', '临床知识', '大学医学']
dev_path = '/cognitive_comp/yangping/data/eval_data/cmmlu/dev.json'
val_path = '/cognitive_comp/yangping/data/eval_data/cmmlu/val.json'
out_val_path = '/cognitive_comp/ganruyi/datasets/medical_data/cmmlu_medical/val.json'
out_dev_path = '/cognitive_comp/ganruyi/datasets/medical_data/cmmlu_medical/dev.json'
def process(input_file, out_file):
    out = open(out_file, 'w')
    with open(input_file, 'r') as file:
        for line in file.readlines():
            data = json.loads(line)
            type = data['type_en']
            if type in tasks:
                out.write(line)
process(val_path, out_val_path)
process(dev_path, out_dev_path)