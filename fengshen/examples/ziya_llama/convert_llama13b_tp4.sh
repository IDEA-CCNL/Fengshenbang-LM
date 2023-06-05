#!/bin/bash
script_path="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/utils/llama_convert/convert_fs_llama_tp.py"
input_dir="/cognitive_comp/ganruyi/fengshenbang-workspace/llama13b_fs"
output_dir="/cognitive_comp/ganruyi/fengshenbang-workspace/llama13b_fs_tp4"
python $script_path --input_dir $input_dir --output_dir $output_dir --model_parallel_size 4