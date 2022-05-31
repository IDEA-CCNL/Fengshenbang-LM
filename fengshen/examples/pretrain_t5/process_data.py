# coding=utf8
from fengshen.data.t5_dataloader.t5_datasets import UnsuperviseT5Dataset
import argparse
import sys

sys.path.append('../../../')

if __name__ == '__main__':
    total_parser = argparse.ArgumentParser("Save data Task")
    total_parser.add_argument(
        '--new_vocab_path', default='/cognitive_comp/ganruyi/hf_models/t5_cn_small/sentencepiece_cn.model', type=str)
    total_parser.add_argument('--preprocessing_num_workers', default=30, type=int)
    total_parser.add_argument(
        '--tokenized_data_path', default='/cognitive_comp/common_data/test_wudao_180g_mt5_tokenized/', type=str)
    total_parser.add_argument('--tokenized_data_shards', default=800, type=int)
    # tokenized_path = '/cognitive_comp/common_data/test_wudao_180g_mt5_tokenized/'
    # tokenized_path = '/cognitive_comp/common_data/wudao_180g_mt5_tokenized/'

    total_parser.add_argument('--max_seq_length', default=512, type=int)
    # * Args for data preprocessing
    args = total_parser.parse_args()
    args.preprocessing_num_workers = 100
    args.tokenized_data_shards = 800
    args.tokenized_data_path = '/cognitive_comp/common_data/wudao_180g_t5_tokenized_512/'
    # ds = UnsuperviseT5Dataset('wudao_280g_test', args)
    ds = UnsuperviseT5Dataset('wudao_180g', args, text_column_name='text', remove_columns=['text'])
    tokenizer = ds.tokenizer
    # 正式数据测试
    # ds = UnsuperviseT5Dataset(
    #     'wudao_180g', args, text_column_name='text', remove_columns=['text'])

    # 测试数据测试
    # ds = UnsuperviseT5Dataset('wudao_180g_mt5_tokenized', args, remove_columns=[], load_data_type=1)
    # print(ds.data, ds.data.features)
    # args.train_data_path = 'wudao_180g_mt5_tokenized'
    # args.preprocessing_num_workers = 1
    # args.train_batchsize = 1
    # dm = UnsuperviseT5DataModel(args)
    # for d in dm.train_dataloader():
    #     print('here\n', flush=True)
