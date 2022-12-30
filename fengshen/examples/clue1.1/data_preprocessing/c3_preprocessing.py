import json
from tqdm import tqdm
import os
import argparse


def load_data(file_path,is_training=False):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = json.loads(''.join(f.readlines()))
        result=[]
        for line in tqdm(lines): 
            data = line
            texta = '\n'.join(data[0])
            textb =''
            for qa in data[1]:
                question=qa['question']
                choice=qa['choice']
                answer=qa['answer'] if 'answer' in qa.keys() else ''
                label = qa['choice'].index(answer) if 'answer' in qa.keys() else 0
                text_id = qa['id'] if 'id' in qa.keys() else 0
                result.append({'texta':texta,
                                'textb':textb,
                                'question':question,
                                'choice':choice,
                                'answer':answer,
                                'label':label,
                                'id':text_id}) 
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
    
    file_list=['d-train','d-dev','c3-m-train','m-train','m-dev','test1.0','test1.1']
    train_data = []
    dev_data = []
    for file in file_list:
        file_path = os.path.join(data_path,file+'.json')
        data=load_data(file_path=file_path)
        if 'train' in file or 'd-dev' in file:
            train_data.extend(data)
        elif 'm-dev' in file:
            dev_data.extend(data)
        elif 'test' in file:
            output_path = os.path.join(save_path,file+'.json')
            save_data(data,output_path)
    
    output_path = os.path.join(save_path,'train.json')
    save_data(train_data,output_path)
            
    output_path = os.path.join(save_path,'dev.json')
    save_data(dev_data,output_path)