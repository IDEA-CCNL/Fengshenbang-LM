#!/bin/bash
script_path="fengshen/utils/llama_convert/convert_fs_llama_tp.py"
input_dir="llama13b_fs"
output_dir="llama13b_fs_tp8"
python $script_path --input_dir $input_dir --output_dir $output_dir --model_parallel_size 8