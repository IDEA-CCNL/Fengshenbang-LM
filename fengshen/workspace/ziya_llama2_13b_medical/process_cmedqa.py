#coding=utf8
import json
cache_dir = '/cognitive_comp/ganruyi/datasets/'
from datasets import load_dataset
dataset = load_dataset("bigbio/med_qa", cache_dir=cache_dir)
med_qa = 'med_qa_zh_source'
train_or_test = 'test'
input_path = 'med_qa_zh_source'
root_dir = '/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data'
output_path = f'{root_dir}/{train_or_test}_{input_path}.json'
out_file = open(output_path, 'w')
for aid, qid in :
    # print(aid, qid)
    out_data = {}
    out_data['prompt'] = [q_dict[qid]]
    out_data['output'] = [a_dict[aid]]
    out_file.write(json.dumps(out_data, ensure_ascii=False))
    out_file.write('\n')




print("Processing complete. Check the output.csv file.")