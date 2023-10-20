#coding=utf8
import json

def convert_to_json(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    json_list = []
    json_dict = {}
    pre_line_type='id'
    pre_p = ''
    pre_d = ''
    for line in lines:
        if line.startswith('id\t'):
            if json_dict:
                if pre_p != '' and pre_line_type == 'P':
                    json_dict['input'].append(pre_p)
                    pre_p = ''
                if pre_d != '' and pre_line_type == 'D':
                    json_dict['output'].append(pre_d)
                    pre_d = ''
                json_list.append(json_dict)
                json_dict = {}

            id = line.strip().split('\t')[1]
            json_dict['id'] = id
            json_dict['input'] = []
            json_dict['output'] = []
            pre_p = pre_d = ''
            pre_line_type = 'id'
        elif line.startswith('P\t'):
            input_text = line.strip().split('\t')[1]
            if pre_line_type == 'D':
                json_dict['output'].append(pre_d)
                pre_d = ''
            pre_p += input_text + '\n'
            pre_line_type = 'P'
            # json_dict['input'].append(input_text)
        elif line.startswith('D\t'):
            output_text = line.strip().split('\t')[1]
            if pre_line_type == 'P':
                json_dict['input'].append(pre_p)
                pre_p = ''
            pre_d += output_text + '\n'
            pre_line_type = 'D'
            # json_dict['output'].append(output_text)
            
    if json_dict:
        json_list.append(json_dict)
    
    return json_list


if __name__ == '__main__':
    input_path = '/cognitive_comp/ganruyi/datasets/medical_data/chunyu/train.dialog'
    output_path = '/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/multi_turn_chimed_train.json'
    # file_path = '../conversations.txt'  # replace with your file path
    json_list = convert_to_json(input_path)
    with open(output_path, 'w') as out:
        for item in json_list:
            json_str = json.dumps(item, ensure_ascii=False)
            out.write(f'{json_str}\n')