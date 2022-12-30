import json
from tqdm import tqdm
import argparse
import numpy as np

def save_data(data,file_path):
    with open(file_path, 'w', encoding='utf8') as f:
        for line in data:
            json_data=json.dumps(line,ensure_ascii=False)
            f.write(json_data+'\n')


def load_data(file_path,is_training=False):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        result=[]
        for l,line in tqdm(enumerate(lines)): 
            data = json.loads(line)
            result.append(data)
        return result


def recls(line):
    mat=[]
    for l in line:
        s=[v for v in l['score'].values()]
        mat.append(s)
    mat=np.array(mat)
    batch,num_labels=mat.shape
    for i in range(len(line)):
        index = np.unravel_index(np.argmax(mat, axis=None), mat.shape)
        line[index[0]]['label'] = int(index[1])
        mat[index[0],:] = np.zeros((num_labels,))
        mat[:,index[1]] = np.zeros((batch,))
    return line

     
import copy                                                                                                                                     

def csl_scorted(data):
    lines={}
    new_data=copy.deepcopy(data)
    for d in data:
        if d['texta'] not in lines.keys():
            lines[d['texta']]={}
        lines[d['texta']][d['id']]=d['score'][d['choice'][0]]
    result=[]
    id2preds={}
    for k,v in lines.items():
        v=sorted(v.items(), key=lambda x: x[1], reverse=True)
        # print(v)
        for i,(text_id, score) in enumerate(v):
            if i<len(v)/2:
                label=0
            else:
                label=1
            id2preds[text_id]=label

    for d in range(len(new_data)):
        new_data[d]['label']=id2preds[new_data[d]['id']]

    return new_data


def submit(file_path):
    id2label={1:'0',0:'1'}
    lines=csl_scorted(load_data(file_path))
    result=[]
    for line in tqdm(lines): 
        data = line
        result.append({'id':data['id'],'label':str(id2label[data['label']])})
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--data_path", type=str,default="")
    parser.add_argument("--save_path", type=str,default="")

    args = parser.parse_args()
    save_data(submit(args.data_path), args.save_path)

    