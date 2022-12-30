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
            result.append({'id':data['id'],'label':data['label']})
    return result

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--data_path", type=str,default="")
    parser.add_argument("--save_path", type=str,default="")

    args = parser.parse_args()

    save_data(submit(args.data_path), args.save_path)
    
    
    