import json
from tqdm import tqdm
import os
import re
import argparse

mask_token='[MASK]'
label_mask='__'


def load_schema(train_answer,dev_answer):
    with open(train_answer,'r',encoding='utf-8') as f:
        train2id = json.loads(''.join(f.readlines()))
    
    with open(dev_answer,'r',encoding='utf-8') as f:
        dev2id = json.loads(''.join(f.readlines()))    
    for k,v in dev2id.items():
        train2id[k]=v

    return train2id
    

def cut(sentence):
    """
	将一段文本切分成多个句子
	:param sentence: ['虽然BillRoper正忙于全新游戏
	:return: ['虽然BillRoper正..接近。' , '与父母，之首。' , '很多..常见。' , '”一位上..推进。' , ''”一直坚..市场。'' , '如今，...的70%。']
	"""
    new_sentence = []
    sen = []
    for i in sentence: # 虽
        sen.append(i)
        if i in ['。', '！', '？', '?',',','，']:
            new_sentence.append("".join(sen)) #['虽然BillRoper正...接近。' , '与父母，...之首。' , ]
            sen = []

    if len(new_sentence) <= 1:  # 一句话超过max_seq_length且没有句号的，用","分割，再长的不考虑了。
        new_sentence = []
        sen = []
        for i in sentence:
            sen.append(i)
            if i.split(' ')[0] in ['，', ','] and len(sen) != 0:
                new_sentence.append("".join(sen))
                sen = []

    if len(sen) > 0:  # 若最后一句话无结尾标点，则加入这句话
        new_sentence.append("".join(sen))
    return new_sentence


def get_answer_text(text,m):
    sent_list=cut(text)
    text1=''
    text2=''
    for i,sent in enumerate(sent_list):
        if m in sent:
            text1=''.join(sent_list[:i])
            if i+1>len(sent_list)-1:
                text2=''
            else:
                text2=''.join(sent_list[i+1:])
            index_text=sent
            return text1,text2,index_text
    return '','',''
    


def load_data(file_path,label2id):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        result=[]
        for l,line in tqdm(enumerate(lines)): 
            data = json.loads(line)
            choice=data['candidates']
            for s,sent in enumerate(data['content']):
                masks=re.findall("#idiom\d{6}#", sent)
                for m in masks:
                    text1,text2,index_text=get_answer_text(sent,m)

                    masks1=re.findall("#idiom\d{6}#", text1)
                    for m1 in masks1:
                        text1=text1.replace(m1,choice[label2id[m1]])
                    
                    masks2=re.findall("#idiom\d{6}#", text2)
                    for m2 in masks2:
                        text2=text2.replace(m2,choice[label2id[m2]])
                        
                    masks3=re.findall("#idiom\d{6}#", index_text)
                    for m3 in masks3:
                        if m3!=m:
                            index_text=index_text.replace(m3,choice[label2id[m3]])

                    choice=[]
                    for c in data['candidates']:
                        choice.append(index_text.replace(m,c))
                        
                    if len('.'.join(choice))>400:
                        choice=data['candidates']
                        text1=text1+index_text.split(m)[0]
                        text2=index_text.split(m)[1]+text2
                        
                    if len(text1)+len(text2)>512-len('.'.join(choice)):
                        split1=0
                        split2=0
                        while split1+split2<512-len('.'.join(choice)):
                            if split1<len(text1):
                                split1+=1
                            if split2<len(text2):
                                split2+=1
                        text1=text1[-split1:]
                        text2=text2[:split2]
                        
                    label=label2id[m] if m in label2id.keys() else 0
                    answer=choice[label] if m in label2id.keys() else ''

                    result.append({'texta':text1,
                                    'textb':text2,
                                    'question':'',
                                    'choice':choice,
                                    'answer':answer,
                                    'label':label,
                                    'id':m,
                                    'text_id':s,
                                    'line_id':l}) 
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
        
    label2id = load_schema(os.path.join(data_path,'train_answer.json'),os.path.join(data_path,'dev_answer.json'))
    
    file_list = ['train','dev','test1.1']
    for file in file_list:
        file_path = os.path.join(data_path,file+'.json')
        output_path = os.path.join(save_path,file+'.json')
        save_data(load_data(file_path,label2id),output_path)
        
        