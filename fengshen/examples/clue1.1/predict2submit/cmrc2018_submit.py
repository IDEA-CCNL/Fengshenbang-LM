import json
from tqdm import tqdm
import argparse


def save_data(data,file_path):
    with open(file_path, 'w', encoding='utf8') as f:
        json_data=json.dumps(data,ensure_ascii=False)
        f.write(json_data+'\n')


def submit(file_path):
    id2score={}
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            for choice in line['choices']:
                if choice['id'] not in id2score.keys():
                    id2score[choice['id']]=[]
                id2score[choice['id']].extend(choice['entity_list'])
        
    result={}
    for k,v in id2score.items():
        if v==[]:
            result[k]=''
        else:
            result[k] = sorted(v, key=lambda k: k['score'],reverse=True)[0]['entity_name']
    return result

 
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--data_path", type=str,default="")
    parser.add_argument("--save_path", type=str,default="")

    args = parser.parse_args()
    save_data(submit(args.data_path), args.save_path)
     
    
    