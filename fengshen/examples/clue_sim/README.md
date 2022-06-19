# 导航
  - [介绍](#介绍)
  - [环境](#环境)
  - [用法](#用法)

## 介绍
clue的语义匹配榜(https://www.cluebenchmarks.com/sim.html)demo

简单的使用二郎神(Erlangshen-MegatronBert)就可以击败其他模型的方法。

## 环境
* Python >= 3.6
* torch == 1.8.0+cu111
* transforms == 4.6.0
* pytorch-lightning == 1.3.2

## 用法
#Train
python -u train.py \
    --mode 'Train' \
    --num_epochs 7 \
    --model_path './weights/Erlangshen-MegatronBert-1.3B-Similarity' \
    --model_name 'IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Similarity'

#Test
python -u train.py \
    --mode 'Test' \
    --num_epochs 7 \
    --model_path './weights/Erlangshen-MegatronBert-1.3B-Similarity' \
    --model_name 'IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Similarity'
