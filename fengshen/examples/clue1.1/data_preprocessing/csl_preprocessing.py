import json
from tqdm import tqdm
import os
import jieba.analyse
import argparse


label2desc={'1':'可以','0':'不能'}

def load_data(file_path,is_training=False):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        result=[]
        for line in tqdm(lines): 
            data = json.loads(line)
            texta = data['abst']
            abst = data['abst']
            textb = ''
            keyword = '、'.join(data['keyword'])
            question = ''


            keyword=data['keyword']
            rs=jieba.analyse.extract_tags(data['abst'],topK=15)
            texta='、'.join(rs)+'。'+texta
            comm=[]
            for k in keyword:
                if k in rs:
                    comm.append(k)

            for word in comm:
                if word in abst:
                    abst=abst.replace(word,word+'（共现关键字）')

            comm=[word for word in comm] 
            keyword=[word for word in data['keyword']] 

            comm_text='共现词汇'+str(len(comm))+'个，分别是'+'、'.join(comm)
            
            keyword = '、'.join(keyword)
            question=''


            choice = [f'{v}使用{keyword}概括摘要' for k,v in label2desc.items()]
            answer = label2desc[data['label']] if 'label' in data.keys() else ''
            answer = f'{answer}使用{keyword}概括摘要'

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
    
    file_list = ['train','dev','test']
    for file in file_list:
        file_path = os.path.join(data_path,file+'.json')
        output_path = os.path.join(save_path,file+'.json')
        save_data(load_data(file_path),output_path)
