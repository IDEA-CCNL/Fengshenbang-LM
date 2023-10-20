#coding=utf8
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM, AutoModelForCausalLM
import torch
import json, argparse, openai, time
from fengshen.models.baichuan.modeling_baichuan import BaichuanForCausalLM
from fengshen.models.baichuan.tokenization_baichuan import BaichuanTokenizer
args_parser = argparse.ArgumentParser()

task_data_dict = {
    'chimed': '/cognitive_comp/ganruyi/Fengshenbang-LM/fengshen/workspace/ziya_llama2_13b_medical/data/chimed.test.json',
    'cmedqa2': '',
}

from fengshen.utils import chinese_char_tokenize
args_parser.add_argument('--task', type=str, default='chimed')
args_parser.add_argument('--dev', type=str, default=None)
args_parser.add_argument('--model_path', type=str, default='/cognitive_comp/yangping/checkpoints/llama/llama2hf/hf_llama13b_step43000')
args_parser.add_argument('--model_name', type=str, default='ziya-v1')
args_parser.add_argument('--save_path', type=str, default='')


args = args_parser.parse_args()

assert args.model_name in ['ziya-v1', 'ziya-v2', 'medicalgpt-ziya', 'medicalgpt-baichuan', 'gpt-3.5-turbo', 'gpt-4', 'bentsao', 'chimed-gpt', 'baichuan']

def call(content: str, max_tokens: int, model='gpt-4'):
    print(content)
    prompt = '假设你是一个专业的医生，下面是一些病人的咨询，请给你非常有用的回答，解决病人的问题，不要让出现让病人去医院就诊或者咨询其他专业医生的建议，不能超过200字。'
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{
                    'role': 'system',
                    'content': prompt,
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except openai.error.RateLimitError as e:
            print(e)
            time.sleep(3)
            pass
        except Exception as e:
            time.sleep(3)
            print(e)
        time.sleep(3)

    print('success!')
    print(response.choices[0].message.content)
    # return response['choices'][0]['message']['content']
    return response.choices[0].message.content

from nltk.translate.bleu_score import corpus_bleu
def compute_bleu(preditions, labels):
    weights = [(0.5,0.5), (0.333,0.333,0.334)]
    score_nltk = corpus_bleu(labels, preditions, weights=weights)
    return score_nltk

def generate(query, args, model, tokenizer, dev=None, few_shot=5):
    prompt = ""
    if dev:
        for i in range(few_shot):
            d=dev[i]
            prompt+=f'<human>:{d["prompt"][0]}\n<bot>:{d["output"][0]}\n'
    # print(f'prompt:{prompt}\n')
    inputs = prompt + '<human>:' + query.strip() + '\n<bot>:'

    input_ids = tokenizer(inputs, return_tensors="pt").input_ids.cuda()
    generate_ids = model.generate(
                input_ids,
                max_new_tokens=512, 
                do_sample = True, 
                top_p = 0.85, 
                temperature = 1.0, 
                repetition_penalty=1., 
                eos_token_id=2, 
                bos_token_id=1, 
                pad_token_id=0)
    output = tokenizer.batch_decode(generate_ids)[0]
    output = output.replace('<bot>  :', '<bot>:').split('<bot>:')[-1]
    print(output)
    return output

if __name__ == '__main__':
    if 'baichuan' in args.model_name or 'Baichuan' in args.model_name:
        model = BaichuanForCausalLM.from_pretrained(args.model_path,device_map="auto",torch_dtype=torch.float16,trust_remote_code=True).half()
        tokenizer = BaichuanTokenizer.from_pretrained(args.model_path,trust_remote_code=True)
    elif args.model_name not in ['gpt-4', 'gpt-3.5-turbo']:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, device_map='auto', trust_remote_code=True)
    
    
    if args.dev:    
        dev = []
        for line in open(args.dev, 'r').readlines():
            dev_data = json.loads(line)
            dev.append(dev_data)
    else:
        dev = None

    input_path = task_data_dict[args.task]
    out_file = open(args.save_path, 'w')
    with open(input_path, 'r') as file:
        predicts = []
        outputs = []
        for line in file.readlines():
            data = json.loads(line)
            prompt = data['prompt'][0]
            output = data['output'][0]
            if args.model_name not in ['gpt-4', 'gpt-3.5-turbo']:
                predict = generate(query=prompt, args=args, model=model, tokenizer=tokenizer, dev=dev)
            else:
                predict = call(prompt, 256, args.model_name)
            predicts.append(chinese_char_tokenize(predict))
            outputs.append(chinese_char_tokenize(output))
            output_json = {'prompt': prompt, 'output': output, 'predict': predict}
            write_str = json.dumps(output_json, ensure_ascii=False)
            out_file.write(f'{write_str}\n')
    # rouge
    from rouge import Rouge
    rouge = Rouge()
    scores = rouge.get_scores(predicts, outputs, avg=True) 
    print(scores)
    # bleu
    outputs = [[o] for o in outputs]
    print(compute_bleu(predicts, outputs))
    out_file.close()