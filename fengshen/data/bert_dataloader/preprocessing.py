import re
import json
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from itertools import chain

_SPLIT_DATA_PATH = '/data1/datas/wudao_180g'


def cut_sent(path):
    """
    中文分句，默认？、。、！、省略号分句，考虑双引号包裹的句子
    采用分割替换的方式
    """
    path = Path(path)
    # print(path)
    save_path = str(Path('/data1/datas/wudao_180g_split',path.name))
    print('处理文件：',save_path)
    with open(save_path,'wt',encoding='utf-8') as w:
        with open(path,'rt',encoding='utf-8') as f:
            for para in tqdm(f):
                para = json.loads(para)
                para_ = para['text']
                # print('sentence piece......')
                para_ = re.sub('([？]+|[。]+|[！]+|[!]+|[…]+|[\.]{3,})([^”’])', r"\1#####\2", para_)
                para_ = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1#####\2', para_)
                # 一个512里面多个样本
                line_ = ''
                for line in para_.split('#####'):
                    line = line.strip()
                    if len(line_) <512:
                        line_ += line
                    else:
                        w.writelines(json.dumps({'text':line_},ensure_ascii=False)+'\n')
                        line_ = line
                w.writelines(json.dumps({'text':line_},ensure_ascii=False)+'\n')

def chain_iter(*filenames):
    """
    将多个文件读成一个迭代器
    """
    reader = [open(file,'r') for file in filenames]
    return chain(*reader)

class Config(object):

    def __init__(self,data_path=_SPLIT_DATA_PATH,num_worker=16,split_numb=600000,cut_sentence=True,output_file=None) -> None:
        self.data_path = Path(data_path)
        self.num_worker = num_worker
        self.split_numb = split_numb
        self.cut_sentence = cut_sentence


def processing1():
    args = Config()
    p_ = [str(i) for i in args.data_path.glob('*')]
    fin = chain_iter(*p_)
    pool = multiprocessing.Pool(args.num_worker)
    docs = pool.imap(cut_sent,fin,chunksize=args.num_worker)
    
    if not Path(args.data_path.parent,args.data_path.name+'_split').exists():
        Path(args.data_path.parent,args.data_path.name+'_split').mkdir()
    writer = open(str(Path(args.data_path.parent,args.data_path.name+'_split','sentence_level.json')),'wt',encoding='utf-8')
    for doc in tqdm(docs):
        for sentence in doc:
            writer.writelines(json.dumps({"text":sentence},ensure_ascii=False)+'\n')
    pool.close()
    pool.join()
    writer.close()


if __name__ == '__main__':
    from time import process_time,perf_counter
    from random import shuffle
    st = process_time()
    args = Config(num_worker=16)

    if not Path(args.data_path.parent,args.data_path.name+'_split').exists():
        Path(args.data_path.parent,args.data_path.name+'_split').mkdir(parents=True)
    
    p_ = [str(i) for i in args.data_path.glob('*')]
    # 简单shuffle
    shuffle(p_)

    pool = multiprocessing.Pool(args.num_worker)
    for item in p_:
        pool.apply_async(func=cut_sent,args=(item,))
    pool.close()
    pool.join()
    cost_time = process_time() - st
    print('DONE!! cost time : %.5f'%cost_time)