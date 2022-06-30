# coding=utf8
import argparse
import sys
import os
from concurrent.futures import ProcessPoolExecutor


def _generate_cache_arrow(index, ds, path):
    print('saving dataset shard {}'.format(index))
    ds.save_to_disk(os.path.join(path, 'part_{}'.format(index)))
    return 'saving dataset shard {} done'.format(index)


def generate_arrow_cache(ds, args) -> None:
    '''
    读取wudao_180g等原数据或者tokenized之后的数据，并进行train test split
    同时利用seed 42做shuffle 缓存下来
    '''
    ds = ds.train_test_split(train_size=args.train_split_size, seed=42)
    print(ds)
    p = ProcessPoolExecutor(max_workers=args.preprocessing_num_workers)
    res = []
    train_shard_part = args.saved_data_shards
    for i in range(0, train_shard_part):
        res.append(p.submit(_generate_cache_arrow, i,
                            ds['train'].shard(train_shard_part, i), args.saved_train_data_path))

    p.shutdown(wait=True)
    for future in res:
        print(future.result(), flush=True)

    ds['test'].save_to_disk(args.saved_test_data_path)
    print('done')


if __name__ == '__main__':
    total_parser = argparse.ArgumentParser("Save data Task")
    total_parser.add_argument(
        '--new_vocab_path', default='/cognitive_comp/ganruyi/hf_models/t5_cn_small/sentencepiece_cn.model', type=str)
    total_parser.add_argument('--preprocessing_num_workers', default=30, type=int)
    total_parser.add_argument(
        '--train_data_path', default='/cognitive_comp/common_data/test_wudao_180g_mt5_tokenized/', type=str)
    total_parser.add_argument('--saved_data_shards', default=800, type=int)
    total_parser.add_argument('--saved_train_data_path', default=None, type=str)
    total_parser.add_argument('--saved_test_data_path', default=None, type=str)
    total_parser.add_argument('--max_seq_length', default=512, type=int)
    total_parser.add_argument('--train_split_size', default=0.999, type=float)
    total_parser.add_argument('--pretrained_model_path', default=None, type=str)
    total_parser.add_argument('--tokenizer_type', default='t5_tokenizer', choices=['t5_tokenizer', 'bert_tokenizer'])
    total_parser.add_argument('--text_column_name', default='text')
    total_parser.add_argument('--remove_columns', nargs='+', default=[])

    # * Args for data preprocessing
    args = total_parser.parse_args()
    sys.path.append('../../../')
    from fengshen.data.t5_dataloader.t5_datasets import UnsuperviseT5Dataset
    ds = UnsuperviseT5Dataset(args.train_data_path, args)
    print(ds)
    generate_arrow_cache(ds.data, args=args)
    # ds = UnsuperviseT5Dataset(args.train_data_path, args, load_data_type=0)
    for i in range(0, 2):
        print(ds.data[i])
        print(ds.tokenizer.decode(ds.data[i]['input_ids']))

    print(ds.data)
