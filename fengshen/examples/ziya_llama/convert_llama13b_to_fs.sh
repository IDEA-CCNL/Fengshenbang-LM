#!/bin/bash
script_path="fengshen/utils/llama_convert/hf_to_fs.py"
input_dir="llama13b_hf"
output_dir="llama13b_fs"
python $script_path --input_path $input_dir --output_path $output_dir