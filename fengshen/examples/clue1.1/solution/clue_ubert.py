import argparse
from fengshen import UbertPipelines
import os
import json
from tqdm import tqdm

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
    total_parser.add_argument('--output_path',default='./predict.json', type=str)
    
    total_parser = UbertPipelines.pipelines_args(total_parser)
    args = total_parser.parse_args()

    train_data = load_data(os.path.join(args.data_dir, args.train_data))
    dev_data = load_data(os.path.join(args.data_dir, args.valid_data))
    test_data = load_data(os.path.join(args.data_dir, args.test_data))
    
    # test_data = test_data[:10]

    model = UbertPipelines(args)
    if args.train:
        model.fit(train_data, dev_data)

    result = model.predict(test_data)
    for line in result[:20]:
        print(line)

    with open(args.output_path, 'w', encoding='utf8') as f:
        for line in result:
            json_data = json.dumps(line, ensure_ascii=False)
            f.write(json_data+'\n')


if __name__ == "__main__":
    main()
