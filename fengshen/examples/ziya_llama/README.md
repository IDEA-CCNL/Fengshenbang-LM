step1 将huggingface版本模型转化成fengshen的格式
```
sh convert_llama13b_to_fs.sh
```

step2 按照显存大小进行tensor并行切分,这里使用8张3090,Tensor parallesim = 8
```
sh convert_llama13b_tp8.sh
```

step3 训练

N=1, TP=4/8 OOM



