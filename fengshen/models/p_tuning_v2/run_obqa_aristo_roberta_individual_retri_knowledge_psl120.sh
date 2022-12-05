export TASK_NAME=superglue
export DATASET_NAME=obqa
export CUDA_VISIBLE_DEVICES=7

bs=4
lr=1e-3
dropout=0.1
psl=120
epoch=240
used_triplets_type=individual_retri

python3 run.py \
  --model_name_or_path LIAMF-USP/aristo-roberta \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 400 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --used_triplets_type $used_triplets_type\
  --output_dir checkpoints/run_obqa_aristo_roberta_individual_retri_knowledge_psl100 \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 12 \
  --save_strategy no \
  --evaluation_strategy epoch \
  --nle_prefix \
  --prefix_fusion_way concat 
