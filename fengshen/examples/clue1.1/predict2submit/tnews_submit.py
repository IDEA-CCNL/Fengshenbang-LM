import json
from tqdm import tqdm
import argparse


def save_data(data,file_path):
    with open(file_path, 'w', encoding='utf8') as f:
        for line in data:
            json_data=json.dumps(line,ensure_ascii=False)
            f.write(json_data+'\n')

def submit(file_path):
    id2label={"故事": "100",
            "文化": "101", 
            "娱乐": "102",
            "体育": "103",
            "财经": "104",
            "房产": "106", 
            "汽车": "107",
            "教育": "108", 
            "科技": "109", 
            "军事": "110", 
            "旅游": "112",
            "国际": "113", 
            "股票": "114",
            "农业": "115",
            "电竞": "116"}
    
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        result=[]
        for line in tqdm(lines): 
            data = json.loads(line)
            result.append({'id':data['id'],'label':id2label[data['choice'][data['label']]]})
    return result


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--data_path", type=str,default="")
    parser.add_argument("--save_path", type=str,default="")

    args = parser.parse_args()
    save_data(submit(args.data_path), args.save_path)
    
    
    