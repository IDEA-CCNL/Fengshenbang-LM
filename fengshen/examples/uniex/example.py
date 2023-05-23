import argparse
from fengshen.pipelines.information_extruction import UniEXPipelines
import os
import json
from tqdm import tqdm
import copy
import time


def load_data(data_path):
    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        samples = [json.loads(line) for line in tqdm(lines)]
    return samples


def main():
    total_parser = argparse.ArgumentParser("TASK NAME")
    total_parser.add_argument('--data_dir', default='./data', type=str)
    total_parser.add_argument('--train_data', default='train.json', type=str)
    total_parser.add_argument('--valid_data', default='dev.json', type=str)
    total_parser.add_argument('--test_data', default='test.json', type=str)
    total_parser = UniEXPipelines.pipelines_args(total_parser)
    args = total_parser.parse_args()

    train_data = load_data(os.path.join(args.data_dir, args.train_data))
    dev_data = load_data(os.path.join(args.data_dir, args.valid_data))
    test_data = load_data(os.path.join(args.data_dir, args.test_data))

    # train_data=train_data[:10]
    test_data=test_data[:100]
    dev_data=dev_data[:10]
    test_data_ori = copy.deepcopy(test_data)
    
    model = UniEXPipelines(args)
    if args.train:
        model.fit(train_data, dev_data,test_data)
    
    start_time=time.time()
    pred_data = model.predict(test_data)
    consum=time.time()-start_time
    print('总共耗费：',consum)
    print('sent/s：',len(test_data)/consum)

    for line in pred_data[:10]:
        print(line)





if __name__ == "__main__":
    main()
