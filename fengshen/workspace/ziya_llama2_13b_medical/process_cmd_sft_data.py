import json
input_path = '/cognitive_comp/ganruyi/datasets/medical_data/medical/finetune/valid_zh_0.json'
output_path = '/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/valid_cmd_sft.json'
out_file = open(output_path, 'w')
with open(input_path, 'r') as file:
    for line in file.readlines():
        output = {}
        data = json.loads(line)
        out_data = {}
        out_data['prompt'] = [data['instruction']]
        out_data['output'] = [data['output']]
        out_file.write(json.dumps(out_data, ensure_ascii=False))
        out_file.write('\n')