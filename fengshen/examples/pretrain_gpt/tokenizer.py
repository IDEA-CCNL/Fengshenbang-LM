
from transformers import GPTNeoXTokenizerFast
# transformer >= 4.20.1

neo_tokenizer = GPTNeoXTokenizerFast.from_pretrained("/cognitive_comp/yangqi/project/FS/Fengshenbang-LM/fengshen/examples/pretrain_gpt/BPETokenizer-Mix-NEOX")


def tokenize(input_code, tokenizer):
    input_ids = tokenizer.encode(input_code)
    decode_str = tokenizer.decode(input_ids)
    return input_ids, decode_str


test_text = ["北京是中国的首都\n北极是 中国  的首都\n┭┮﹏┭┮_特殊符号\nGPT is a transformers model\nimport tensorflow as tf\ndef __init__(self, model_dir):\n\tfor i in range(0,10):\n\t\tprint(i)\ndef min_seq_len():\n\tans=min(l)\n        这里是8个空格"]
for inputs in test_text:
    input_ids, decode_str = tokenize(inputs, neo_tokenizer)
    print(input_ids)
    print(decode_str)
