import json
from tqdm import tqdm
import argparse

label2desc={"news_story": "故事",
              "news_culture": "文化",
              "news_entertainment": "娱乐",
              "news_sports": "体育",
              "news_finance": "财经",
              "news_house": "房产",
              "news_car": "汽车",
              "news_edu": "教育",
              "news_tech": "科技",
              "news_military": "军事",
              "news_travel": "旅游",
              "news_world": "国际",
              "news_stock": "股票",
              "news_agriculture": "农业",
              "news_game": "电竞"}

def load_data(file_path,is_training=False):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        result=[]
        for line in tqdm(lines): 
            data = json.loads(line)
            texta = data['sentence']
            textb = ''
            question = '下面新闻属于哪一个类别？'
            choice = [v for k,v in label2desc.items()]
            answer = label2desc[data['label_desc']] if 'label_desc' in data.keys() else ''
            label = choice.index(answer) if 'label_desc' in data.keys() else 0
            text_id = data['id'] if 'id' in data.keys() else 0
            result.append({'texta':texta,
                            'textb':textb,
                            'question':question,
                            'choice':choice,
                            'answer':answer,
                            'label':label,
                            'id':text_id}) 
        print(result[0])
        return result


def save_data(data,file_path):
    with open(file_path, 'w', encoding='utf8') as f:
        for line in data:
            json_data=json.dumps(line,ensure_ascii=False)
            f.write(json_data+'\n')

import os

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--data_path", type=str,default="")
    parser.add_argument("--save_path", type=str,default="")

    args = parser.parse_args()
    
    
    data_path = args.data_path
    save_path = args.save_path
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    file_list = ['train','dev','test1.0','test1.1']
    for file in file_list:
        file_path = os.path.join(data_path,file+'.json')
        output_path = os.path.join(save_path,file+'.json')
        save_data(load_data(file_path),output_path)