# ====================================================
#   Copyright (C) 2022  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : convert_zero_to_fp32.py
#   Last Modified : 2022-03-21 21:51
#   Describe      : 
#
# ====================================================


import os
import argparse
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict


# lightning deepspeed has saved a directory instead of a file
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="path to the desired checkpoint folder, e.g., path/checkpoint-12")
    parser.add_argument(
        "--output_file",
        default="pytorch_model.bin",
        type=str,
        help="path to the pytorch fp32 state_dict output file (e.g. path/checkpoint-12/pytorch_model.bin)"
    )
    args = parser.parse_args()

    convert_zero_checkpoint_to_fp32_state_dict(args.checkpoint_dir, os.path.join(args.checkpoint_dir, args.output_file))

