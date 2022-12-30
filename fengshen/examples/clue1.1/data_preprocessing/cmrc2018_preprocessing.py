import json
from tqdm import tqdm
import os
from sklearn.utils import shuffle
import re
import argparse


def cut_sent(para):
    para = re.sub('([。，,！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里
    return para.split("\n")


def search(pattern, sequence):
    n = len(pattern)
    res=[]
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            res.append([i,i + n-1])
    return res

max_length=512
stride=128
def stride_split(question, context, answer, start):
    end = start + len(answer) -1
    results, n = [], 0
    max_c_len = max_length - len(question) - 3
    while True:
        left, right = n * stride, n * stride + max_c_len
        if left <= start < end <= right:
            results.append((question, context[left:right], answer, start - left, end - left))
        elif right < start or end < right:
            results.append((question, context[left:right], '', -1, -1))
        if right >= len(context):
            return results
        n += 1


def load_data(file_path,is_training=False):
    task_type='抽取任务'
    subtask_type='抽取式阅读理解'
    with open(file_path, 'r', encoding='utf8') as f:
        lines = json.loads(''.join(f.readlines()))
        result=[]
        lines = lines['data']
        for line in tqdm(lines): 
            if line['paragraphs']==[]:
                continue
            data = line['paragraphs'][0]
            context=data['context'].strip()
            for qa in data['qas']:
                question=qa['question'].strip()
                rcv=[]
                for a in qa['answers']:
                    if a not in rcv:
                        rcv.append(a)
                        split=stride_split(question, context, a['text'], a['answer_start'])
                        for sp in split:
                            choices = []
                            
                            choice = {}
                            choice['id']=qa['id']
                            choice['entity_type'] = qa['question']
                            choice['label']=0
                            entity_list=[]
                            if sp[3]>=0 and sp[4]>=0:
                                entity_list.append({'entity_name':sp[2],'entity_type':'','entity_idx':[[sp[3],sp[4]]]})
                                
                            choice['entity_list']=entity_list
                            choices.append(choice)
                                
                            if choices==[]:
                                print(data)
                                continue
                            result.append({ 'task_type':task_type,
                                            'subtask_type':subtask_type,
                                            'text':sp[1],
                                            'choices':choices,
                                            'id':0}) 
                            
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
    
    file_list=['dev','train','trial','test']
    train_data = []
    dev_data = []
    for file in file_list:
        file_path = os.path.join(data_path,file+'.json')
        data=load_data(file_path=file_path)
        if 'train' in file or 'trial' in file:
            train_data.extend(data)
        else:
            output_path = os.path.join(save_path,file+'.json')
            save_data(data,output_path)
    
    output_path = os.path.join(save_path,'train.json')
    save_data(train_data,output_path)
            