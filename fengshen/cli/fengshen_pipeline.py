import sys
from importlib import import_module
from datasets import load_dataset
import argparse


def main():
    if len(sys.argv) < 3:
        raise Exception(
            'args len < 3, example: fengshen_pipeline text_classification predict xxxxx')
    pipeline_name = sys.argv[1]
    method = sys.argv[2]
    pipeline_class = getattr(import_module('fengshen.pipelines.' + pipeline_name), 'Pipeline')

    total_parser = argparse.ArgumentParser("FengShen Pipeline")
    total_parser.add_argument('--model', default='', type=str)
    total_parser.add_argument('--datasets', default='', type=str)
    total_parser.add_argument('--text', default='', type=str)
    total_parser = pipeline_class.add_pipeline_specific_args(total_parser)
    args = total_parser.parse_args(args=sys.argv[3:])
    pipeline = pipeline_class(args=args, model=args.model)

    if method == 'predict':
        print(pipeline(args.text))
    elif method == 'train':
        datasets = load_dataset(args.datasets)
        pipeline.train(datasets)
    else:
        raise Exception(
            'cmd not support, now only support {predict, train}')


if __name__ == '__main__':
    main()
