import time
from builtins import print
import argparse

import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def get_time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def main():
    total_parser = argparse.ArgumentParser("Pretrain Unsupervise.")
    total_parser.add_argument('--ckpt_path', default=None, type=str)
    total_parser.add_argument('--bin_path', default=None, type=str)
    total_parser.add_argument('--rm_prefix', default=None, type=str)
    # * Args for base model
    args = total_parser.parse_args()
    print('Argument parse success.')
    state_dict = torch.load(args.ckpt_path)['module']
    new_state_dict = {}

    if args.rm_prefix is not None:
        prefix_len = len(args.rm_prefix)
        for k, v in state_dict.items():
            if k[:prefix_len] == args.rm_prefix:
                new_state_dict[k[prefix_len:]] = v
            else:
                new_state_dict[k] = v
    else:
        new_state_dict = state_dict
    torch.save(new_state_dict, args.bin_path)


if __name__ == '__main__':
    main()
