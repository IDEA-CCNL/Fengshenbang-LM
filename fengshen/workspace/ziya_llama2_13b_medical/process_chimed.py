#coding=utf8
import json
rewrite_dict = {
    'antinuclearantibody': 'antinuclear antibody',
    'hepaticinsufficiency': 'hepatic insufficiency',
    'hepaticencephalopathy': 'hepatic encephalopathy',
    'primarycarcinomaoftheliver': 'primary carcinoma of the liver',
    'chronicglomerulonephritis': 'chronic glomerulonephritis',
    'disseminatedintravascularcoagulation': 'disseminated intravascular coagulation',
    'VitaminCdeficiency': ' Vitamin C deficiency',
    'fattyliver': 'fatty liver',
    'HCVRNA': 'HCV RNA',
    'nephroticsyndrome': 'nephrotic syndrome',
    'disseminatedintravascularcoagulation': 'disseminated intravascular coagulation',
    'guidedboneregeneration': 'guided bonere generation',
    'primaryaldosteronism': 'primary aldosteronism',
    'bodymassindex': 'body mass index',
}
import re
def filter_answer(answer):
    index = answer.find('提问者对于答案的评价')
    return answer[:index] if index != -1 else answer

drop_prefixs = [
    '提问者对于答案的评价',
    '补充问题',
    '提问人的追问',
    '补充提问：',
]
invalid_substrs = [
    'http:',
    'www.',
    'askhpww42013',
    '````````````````',
    '发表时间：',
    'ask7KBKS',
    '电话',
    '热线',
]
invalid_substrs = [substr.lower() for substr in invalid_substrs]
def drop_text_with_prefix(text, prefix):
    index = text.find(prefix)
    return text[:index] if index != -1 else text

def _has_invalid_substr(text, substr):
    return text.find(substr) != -1

def has_invalid_substr(text):
    text = text.lower()
    for substr in invalid_substrs:
        if _has_invalid_substr(text, substr):
            return True
    return False

def replace_long_english(text):
    ret = text
    for key,value in rewrite_dict.items():
        if ret.find(key) != -1:
            ret = ret.replace(key, value)
    return ret

def process(input_path, output_path):
    out_file = open(output_path, 'w')
    exceptions = 0
    with open(input_path, encoding='utf8', errors='ignore') as file:
        for line in file.readlines():
            try:
                out_data = {}
                data = json.loads(line)
                question = data['question']
                answer_1 = data['answer_1'] 
                answer_2 = data['answer_2']
                specialty_1 = data['specialty_1']
                specialty_2 = data['specialty_2']
                adopted_1 = data['adopted_1']
                adopted_2 = data['adopted_2']
                if has_invalid_substr(question) or has_invalid_substr(answer_1) or has_invalid_substr(answer_2):
                    continue
                # filter 提问者对于答案的评价
                for prefix in drop_prefixs:
                    question = drop_text_with_prefix(question, prefix)
                    answer_1 = drop_text_with_prefix(answer_1, prefix)
                    answer_2 = drop_text_with_prefix(answer_2, prefix)
                question = replace_long_english(question)
                answer_1 = replace_long_english(answer_1)
                answer_2 = replace_long_english(answer_2)
                # if adopted_1 == 'true' and adopted_2 == 'false':
                #     out_data['prompt'] = [question]
                #     out_data['output'] = [answer_1]
                #     out_file.write(json.dumps(out_data, ensure_ascii=False))
                #     out_file.write('\n')
                # elif adopted_1 == 'false' and adopted_2 == 'true':
                #     out_data['prompt'] = [question]
                #     out_data['output'] = [answer_2]
                #     out_file.write(json.dumps(out_data, ensure_ascii=False))
                #     out_file.write('\n')
                # else:
                out_data['prompt'] = [question]
                out_data['output'] = [answer_1]
                out_file.write(json.dumps(out_data, ensure_ascii=False))
                out_file.write('\n')
                out_data = {}
                out_data['prompt'] = [question]
                out_data['output'] = [answer_2]
                out_file.write(json.dumps(out_data, ensure_ascii=False))
                out_file.write('\n')
            except:
                # print(f"{exceptions}: {line}")
                exceptions += 1
            finally:
                pass
                # print(f'error line: {exceptions}')


if __name__ == '__main__':
    input_path = '/cognitive_comp/ganruyi/datasets/medical_data/ChiMed_Data_v1/ChiMed.v1.test.json'
    output_path = '/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/chimed.test.json'
    process(input_path, output_path)