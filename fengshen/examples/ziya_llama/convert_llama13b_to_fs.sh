#!/bin/bash
script_path="fengshen/utils/llama_convert/hf_to_fs.py"
input_dir="/cognitive_comp/yangping/checkpoints/llama/llama2hf/hf_llama13b_step43000"
output_dir="/cognitive_comp/ganruyi/fengshenbang-workspace/llama13b_fs"
python $script_path --input_path $input_dir --output_path $output_dir