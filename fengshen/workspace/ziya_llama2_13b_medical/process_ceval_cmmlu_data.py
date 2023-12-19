import json

abc=['A','B','C','D','E','F','G','H']
def get_choice_text(choice):
    
    choice_text=''
    for i,c in enumerate(choice):
        choice_text += abc[i]+'.'+str(c)+'\n'
    return choice_text

def get_prompt2(query,dev=None,few_shot=5):
    prompt=f'[human]:下面是一道关于{query["type"]}的单选题，请选择正确的答案选项。\n问题：'
    prompt+= str(query['text'])+'\n\n'+get_choice_text(query['choice']) + '[bot]:'
    return prompt

def get_ans(query):
    choice = query['choice']
    label = query['label']
    ans = f'{abc[label]}.{choice[label]}'
    return ans

import json
input_path = '/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/ceval_cmmlu.json'
output_path = '/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/train_ceval_cmmlu.json'
out_file = open(output_path, 'w')
with open(input_path, 'r') as file:
    for line in file.readlines():
        output = {}
        data = json.loads(line)
        out_data = {}
        out_data['prompt'] = [get_prompt2(data)]
        out_data['output'] = [get_ans(data)]
        out_file.write(json.dumps(out_data, ensure_ascii=False))
        out_file.write('\n')