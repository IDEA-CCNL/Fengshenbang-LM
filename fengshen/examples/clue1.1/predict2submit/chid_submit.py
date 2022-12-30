import json
from tqdm import tqdm
import argparse
import numpy as np

def save_data(data,file_path):
    with open(file_path, 'w', encoding='utf8') as f:
        json_data=json.dumps(data,ensure_ascii=False)
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

        
        
     
                                                                                                                                     

def chid_m(data):
    lines={}
    for d in data:
        if d['line_id'] not in lines.keys():
            lines[d['line_id']]=[]
        lines[d['line_id']].append(d)
    result=[]
    for k,v in lines.items():
        result.extend(recls(v))
    return result



def submit(file_path):
    lines = chid_m(load_data(file_path))
    result={}
    for line in tqdm(lines): 
        data = line
        result[data['id']]=data['label']
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--data_path", type=str,default="")
    parser.add_argument("--save_path", type=str,default="")

    args = parser.parse_args()
    save_data(submit(args.data_path), args.save_path)
    
    
    