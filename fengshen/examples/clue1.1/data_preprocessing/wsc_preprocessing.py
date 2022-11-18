import json
from tqdm import tqdm
import os
import argparse

label2desc={'true':'是','false':'不是'}


def load_data(file_path,is_training=False):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        result=[]
        for line in tqdm(lines): 
            data = json.loads(line)
            target = data['target']
            text=list(data['text'])
            if target['span2_index']<target['span1_index']:
                text.insert(target['span2_index'],'_')
                text.insert(target['span2_index']+len(target['span2_text'])+1,'_')
                text.insert(target['span1_index']+2,'[')
                text.insert(target['span1_index']+2+len(target['span1_text'])+1,']')
            else:
                text.insert(target['span1_index'],'[')
                text.insert(target['span1_index']+len(target['span1_text'])+1,']')
                text.insert(target['span2_index']+2,'_')
                text.insert(target['span2_index']+2+len(target['span2_text'])+1,'_')

            texta = ''.join(text)
            textb = ''
            span2_text=target['span2_text']
            span1_text=target['span1_text']

            question = ''

            choice = []
            for k,v in label2desc.items():
                choice .append(f'{span2_text}{v}{span1_text}')
            # print(choice)
            answer = label2desc[data['label']] if 'label' in data.keys() else ''
            answer = f'{span2_text}{answer}{span1_text}'

            label = choice.index(answer) if 'label' in data.keys() else 0
            text_id = data['id'] if 'id' in data.keys() else 0
            result.append({'texta':texta,
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
    
    file_list = ['train','dev','test1.0','test1.1']
    for file in file_list:
        file_path = os.path.join(data_path,file+'.json')
        output_path = os.path.join(save_path,file+'.json')
        save_data(load_data(file_path),output_path)