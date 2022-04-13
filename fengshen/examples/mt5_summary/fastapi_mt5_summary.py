import os
import sys
import uvicorn
import torch
from fastapi import Body, FastAPI
from transformers import T5Tokenizer, MT5ForConditionalGeneration
import pytorch_lightning as pl
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
os.environ["MASTER_ADDR"] = '127.0.0.1'
os.environ["MASTER_PORT"] = '6000'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('device')
pretrain_model_path = '/cognitive_comp/ganruyi/hf_models/google/mt5-large'
# pretrain_model_path = 'google/mt5-small'
model_path = '/cognitive_comp/ganruyi/fengshen/mt5_large_summary/ckpt/epoch-0-last.ckpt'
tokenizer = T5Tokenizer.from_pretrained(pretrain_model_path)
print('load tokenizer')


class MT5FinetuneSummary(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained(pretrain_model_path)


model = MT5FinetuneSummary.load_from_checkpoint(model_path)
print('load checkpoint')
model.to(device)
model.eval()
app = FastAPI()
print('server start')

# def flask_gen(text: str, level: float = 0.9, n_sample: int = 5, length: int = 32, is_beam_search=False):


@app.post('/mt5_summary')
async def flask_gen(text: str = Body('', title='原文', embed=True),
                    n_sample: int = 5, length: int = 32, is_beam_search=False):
    if len(text) > 128:
        text = text[:128]
    text = 'summary:'+text
    print(text)
    # inputs = tokenizer(text, return_tensors='pt')
    inputs = tokenizer.encode_plus(
        text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    # print(inputs)
    if is_beam_search:
        generated_ids = model.model.generate(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            max_length=length,
            num_beams=n_sample,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            num_return_sequences=n_sample
        )
    else:
        generated_ids = model.model.generate(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            max_length=length,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=2.5,
            # early_stopping=True,
            num_return_sequences=n_sample
        )
    result = []
    # print(tokenizer.all_special_tokens)
    for sample in generated_ids:
        preds = [tokenizer.decode(sample, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True)]
        preds = ''.join(preds)
        # print(preds)
        result.append(preds)
    return result


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=6607, log_level="debug")
# #     article = "日前，方舟子发文直指林志颖旗下爱碧丽推销假保健品，引起哗然。调查发现，
# 爱碧丽没有自己的生产加工厂。其胶原蛋白饮品无核心研发，全部代工生产。号称有“逆生长”功效的爱碧丽“梦幻奇迹限量组”售价>高达1080元，实际成本仅为每瓶4元！"
#     article = '''在北京冬奥会自由式滑雪女子坡面障碍技巧决赛中，中国选手谷爱凌夺得银牌。祝贺谷爱凌！
# 今天上午，自由式滑雪女子坡面障碍技巧决赛举行。决赛分三轮进行，取选手最佳成绩排名决出奖牌。
# 第一跳，中国选手谷爱凌获得69.90分。在12位选手中排名第三。完成动作后，谷爱凌又扮了个鬼脸，甚是可爱。
# 第二轮中，谷爱凌在道具区第三个障碍处失误，落地时摔倒。获得16.98分。网友：摔倒了也没关系，继续加油！
# 在第二跳失误摔倒的情况下，谷爱凌顶住压力，第三跳稳稳发挥，流畅落地！获得86.23分！此轮比赛，共12位选手参赛，谷爱凌第10位出场。网友：看比赛时我比谷爱凌紧张，加油！'''
    # flask_gen(article, length=30)
