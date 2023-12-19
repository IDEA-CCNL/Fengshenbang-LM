#!/bin/bash
## add mc data
# 源文件夹路径数组
ROOT_DIR="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data"
SRC_DIRS=("${ROOT_DIR}/medical_multiple_choice_data" "${ROOT_DIR}/medical_qa_data" "${ROOT_DIR}/medical_single_choice_data" "${ROOT_DIR}/medical_trueorfalse_data")

# 输出文件的路径
OUTPUT_FILE="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/train_data_v5.json"

# 确保输出文件不存在，如果存在则删除它
[ -f "$OUTPUT_FILE" ] && rm "$OUTPUT_FILE"

# 遍历每个源文件夹
for SRC_DIR in "${SRC_DIRS[@]}"; do
    # 遍历当前文件夹中的所有文件
    for file in $(ls "$SRC_DIR" | sort); do
        # 将文件的内容追加到输出文件中
        cat "$SRC_DIR/$file" >> "$OUTPUT_FILE"
    done
done

sed -i 's/：\\n\\n题目：/。\\n\\n问题：/g' "$OUTPUT_FILE"

echo "All files merged and content replaced in $OUTPUT_FILE"

## add train_cmmlu_ceval data

cmmlu_ceval="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/train_ceval_cmmlu.json"

sed -i 's/[human]://g' "$cmmlu_ceval"
sed -i 's/[bot]://g' "$cmmlu_ceval"

cat "$cmmlu_ceval" >> "$OUTPUT_FILE"

## add chimed data
chimed_data="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/chimed.trainrm_scored_top10w.json"
cat "$chimed_data" >> "$OUTPUT_FILE"

## add multi-turn data
multi_turn_data="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/multi_turn_chimed_train.json"
# 留最后100行作为test集合
train_multi_turn_data="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/train_multi_turn_chimed.json"
test_multi_turn_data="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/test_multi_turn_chimed.json"
# 获取总行数
TOTAL_LINES=$(wc -l < "$multi_turn_data")

# 如果总行数少于100行，脚本将不进行分割
if [[ $TOTAL_LINES -lt 100 ]]; then
    echo "The file has less than 100 lines. Exiting without splitting."
    exit 1
fi

# 计算第一个文件的行数
FIRST_FILE_LINES=$((TOTAL_LINES - 100))

# 使用head和tail命令分割文件
head -n $FIRST_FILE_LINES "$multi_turn_data" > "$train_multi_turn_data"
tail -n 100 "$multi_turn_data" > "$test_multi_turn_data"
echo "Split complete. Check $train_multi_turn_data and $test_multi_turn_data"
cat "$train_multi_turn_data" >> "$OUTPUT_FILE"

## add cmd data
cmd_data="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/train_cmd_sft.json"
head -n 100000 "$cmd_data" >> "$OUTPUT_FILE"

## add ner data
ner_data="/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/chimst.train.json"
cat "$ner_data" >> "$OUTPUT_FILE"

## add cmedqa_v2 data
cmedqa_v2_data="/cognitive_comp/ganruyi/datasets/medical_data/cMedQA2/train_processed.json"
cat "$cmedqa_v2_data" >> "$OUTPUT_FILE"