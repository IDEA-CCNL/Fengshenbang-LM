#!/bin/bash
## add mc data
# 源文件夹路径数组
ROOT_DIR="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data"

# 输出文件的路径
OUTPUT_FILE="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/train_data_v6.json"
OLD_FILE="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/train_data_v5.json"

# 确保输出文件不存在，如果存在则删除它
[ -f "$OUTPUT_FILE" ] && rm "$OUTPUT_FILE"

## add ner data
ner_data="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/med_llm_ner_data/ccks2019-data/ccks2019.train.json"
cat "$ner_data" >> "$OUTPUT_FILE"
cat "$ner_data" >> "$OUTPUT_FILE"
cat "$ner_data" >> "$OUTPUT_FILE"

## add old data 
shuf -n3000 "$OLD_FILE" >> "$OUTPUT_FILE"