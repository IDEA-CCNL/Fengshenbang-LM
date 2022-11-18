import argparse
from fengshen.pipelines.multiplechoice import UniMCPipelines
import os
import json
import copy
from tqdm import tqdm

def load_data(data_path):
    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        samples = [json.loads(line) for line in tqdm(lines)]
    return samples


def comp_acc(pred_data,test_data):
    corr=0
    for i in range(len(pred_data)):
        if pred_data[i]['label']==test_data[i]['label']:
            corr+=1
    return corr/len(pred_data)


def main():
    total_parser = argparse.ArgumentParser("TASK NAME")
    total_parser.add_argument('--data_dir', default='./data', type=str)
    total_parser.add_argument('--train_data', default='train.json', type=str)
    total_parser.add_argument('--valid_data', default='dev.json', type=str)
    total_parser.add_argument('--test_data', default='test.json', type=str)
    total_parser.add_argument('--output_path', default='', type=str)
    
    total_parser = UniMCPipelines.piplines_args(total_parser)
    args = total_parser.parse_args()

    train_data = load_data(os.path.join(args.data_dir, args.train_data))
    dev_data = load_data(os.path.join(args.data_dir, args.valid_data))
    test_data = load_data(os.path.join(args.data_dir, args.test_data))

    # dev_data = dev_data[:200]
    dev_data_ori=copy.deepcopy(dev_data)

    model = UniMCPipelines(args, args.pretrained_model_path)
    
    print(args.data_dir)
            
    if args.train:
        model.train(train_data, dev_data)
    result = model.predict(dev_data)
    for line in result[:20]:
        print(line)

    acc=comp_acc(result,dev_data_ori)
    print('acc:',acc)
    
    if args.output_path != '':
        test_result = model.predict(test_data)
        with open(args.output_path, 'w', encoding='utf8') as f:
            for line in test_result:
                json_data=json.dumps(line,ensure_ascii=False)
                f.write(json_data+'\n')


if __name__ == "__main__":
    main()
