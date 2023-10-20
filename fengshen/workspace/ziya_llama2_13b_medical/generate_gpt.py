#coding=utf8
import torch
import json, argparse, openai, time
args_parser = argparse.ArgumentParser()

task_data_dict = {
    'chimed': 'chimed.test.json',
    'cmedqa2': '',
}

args_parser.add_argument('--task', type=str, default='chimed')
args_parser.add_argument('--dev', type=str, default=None)
args_parser.add_argument('--model_path', type=str, default='/cognitive_comp/yangping/checkpoints/llama/llama2hf/hf_llama13b_step43000')
args_parser.add_argument('--model_name', type=str, default='ziya-v1')
args_parser.add_argument('--save_path', type=str, default='')


args = args_parser.parse_args()


_UCODE_RANGES = (
    ("\u3400", "\u4db5"),  # CJK Unified Ideographs Extension A, release 3.0
    ("\u4e00", "\u9fa5"),  # CJK Unified Ideographs, release 1.1
    ("\u9fa6", "\u9fbb"),  # CJK Unified Ideographs, release 4.1
    ("\uf900", "\ufa2d"),  # CJK Compatibility Ideographs, release 1.1
    ("\ufa30", "\ufa6a"),  # CJK Compatibility Ideographs, release 3.2
    ("\ufa70", "\ufad9"),  # CJK Compatibility Ideographs, release 4.1
    ("\u20000", "\u2a6d6"),  # (UTF16) CJK Unified Ideographs Extension B, release 3.1
    ("\u2f800", "\u2fa1d"),  # (UTF16) CJK Compatibility Supplement, release 3.1
    ("\uff00", "\uffef"),  # Full width ASCII, full width of English punctuation,
    # half width Katakana, half wide half width kana, Korean alphabet
    ("\u2e80", "\u2eff"),  # CJK Radicals Supplement
    ("\u3000", "\u303f"),  # CJK punctuation mark
    ("\u31c0", "\u31ef"),  # CJK stroke
    ("\u2f00", "\u2fdf"),  # Kangxi Radicals
    ("\u2ff0", "\u2fff"),  # Chinese character structure
    ("\u3100", "\u312f"),  # Phonetic symbols
    ("\u31a0", "\u31bf"),  # Phonetic symbols (Taiwanese and Hakka expansion)
    ("\ufe10", "\ufe1f"),
    ("\ufe30", "\ufe4f"),
    ("\u2600", "\u26ff"),
    ("\u2700", "\u27bf"),
    ("\u3200", "\u32ff"),
    ("\u3300", "\u33ff"),
)


def is_chinese_char(uchar):
    for start, end in _UCODE_RANGES:
        if start <= uchar <= end:
            return True
    return False

def chinese_char_tokenize(line):
    line = line.strip()
    line_in_chars = ""

    for char in line:
        if is_chinese_char(char):
            line_in_chars += " "
            line_in_chars += char
            line_in_chars += " "
        else:
            line_in_chars += char

    return line_in_chars


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
    output = output.split('<bot>  :')[-1]
    print(output)
    return output

if __name__ == '__main__':
    if args.dev:    
        dev = []
        for line in open(args.dev, 'r').readlines():
            dev_data = json.loads(line)
            dev.append(dev_data)

    input_path = task_data_dict[args.task]
    out_file = open(args.save_path, 'w')
    with open(input_path, 'r') as file:
        predicts = []
        outputs = []
        for line in file.readlines():
            data = json.loads(line)
            prompt = data['prompt'][0]
            output = data['output'][0]
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