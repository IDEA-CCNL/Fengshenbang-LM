import json
from tqdm import tqdm
import argparse


def save_data(data,file_path):
    with open(file_path, 'w', encoding='utf8') as f:
        for line in data:
            json_data=json.dumps(line,ensure_ascii=False)
            f.write(json_data+'\n')

def submit(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        result=[]
        for line in tqdm(lines): 
            data = json.loads(line)
            if '不是' in data['choice'][0] and '是' in data['choice'][1]:
                if data['label']==1:
                    label='false'
                else:
                    label='true'
            else:
                if data['label']==0:
                    label='true'
                else:
                    label='false'
            result.append({'id':data['id'],'label':label})
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--data_path", type=str,default="")
    parser.add_argument("--save_path", type=str,default="")

    args = parser.parse_args()
    save_data(submit(args.data_path), args.save_path)
    
    
    