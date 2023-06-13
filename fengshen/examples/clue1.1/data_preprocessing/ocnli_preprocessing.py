import json
from tqdm import tqdm
import os
import argparse


label2desc={'contradiction':'矛盾','neutral':'自然','entailment':'蕴含'}

def load_data(file_path,is_training=False):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        result=[]
        for line in tqdm(lines): 
            data = json.loads(line)
            texta = data['sentence1']
            textb = data['sentence2']
            question = ''
            choice = [v for k,v in label2desc.items()]
            answer = label2desc[data['label']] if 'label' in data.keys() else ''
            label = choice.index(answer) if 'label' in data.keys() else 0
            text_id = data['id'] if 'id' in data.keys() else 0
            result.append({'task_type':'自然语言推理',
                           'texta':texta,
                            'textb':textb,
                            'question':question,
                            'choice':choice,
                            'answer':answer,
                            'label':label,
                            'id':text_id}) 
        for i in range(5):
            print(result[i])
        return result


def save_data(data,file_path):
    with open(file_path, 'w', encoding='utf8') as f:
        for line in data:
            json_data=json.dumps(line,ensure_ascii=False)
            f.write(json_data+'\n')


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--data_path", type=str,default="")
    parser.add_argument("--save_path", type=str,default="")

    args = parser.parse_args()
    
    
    data_path = args.data_path
    save_path = args.save_path
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    file_list = ['train','dev','test']
    for file in file_list:
        file_path = os.path.join(data_path,file+'.json')
        output_path = os.path.join(save_path,file+'.json')
        save_data(load_data(file_path),output_path)
