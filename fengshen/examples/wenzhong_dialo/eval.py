import argparse
from collections import Counter
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os, random, re
import streamlit as st
import torch
import json
import jieba
from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestModule:
    def __init__(self, **args):
        self.args = argparse.Namespace(**args)
        self.model_id = self.args.model_id
        self.tokenizer, self.model = self.load_model(self.args.model_id)

    @staticmethod
    def set_seed(seed=42):
        random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def load_model(self, model_id):
        if model_id == 'Wenzhong-Finetune-110M-Loss0.7':
            #model_path = "/cognitive_comp/yangqi/logs/wenzhong_dialo/ckpt/hf_pretrained_epoch3_step3906"
            model_path = "/cognitive_comp/yangqi/logs/wenzhong_dialo/Wenzhong-GPT2-110M/ckpt/hf_pretrained_model"
        elif model_id == 'Wenzhong-Finetune-Loss0.39' or "Wenzhong-Finetune-Loss0.2" or "Wenzhong-Finetune-Loss0.1" or "Wenzhong-Finetune-Query-Loss0.1":
            model_path = f"/cognitive_comp/yangqi/model/{model_id}"
        elif model_id == "Wenzhong-GPT2-3.5B" or "Wenzhong-GPT2-110M":
            model_path = f"/cognitive_comp/yangqi/model/{model_id}"
            token_path = f"/cognitive_comp/yangqi/model/{model_id}"
        else: 
            model_path = None
        token_path = "/cognitive_comp/yangqi/model/Wenzhong-GPT2-110M"

        if model_path is None:
            model_path = "/cognitive_comp/yangqi/model/Wenzhong-GPT2-110M"

        tokenizer = GPT2Tokenizer.from_pretrained(token_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)

        model.to(device)

        # tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # [PAD]
        # tokenizer.add_special_tokens({"bos_token": "<s>"})  # <s>
        # tokenizer.add_special_tokens({"eos_token": "</s>"})  # </s>
        # tokenizer.add_special_tokens({"unk_token": "<unk>"})  # <unk>]
        tokenizer.eos_token = '[SEP]'
        tokenizer.pad_token = '[PAD]'

        return tokenizer, model
    
    def load_from_json(self, data_file):
        examples = []
        with open(data_file,'r',encoding='utf8')as fp:
            while True:
                line = fp.readline()
                if not line: #EOF
                    break
                s = json.loads(line)
                examples.append(s)
        return examples

    @staticmethod
    def truncate(document:str, max_num_tokens:int):
        total_length = len(document)
        if total_length <= max_num_tokens:
            return document
        else: 
            return document[:max_num_tokens]

    def preprocess(self): # need rewrite with each task
        raise NotImplementedError

    def generate(self,input_dict,prompt):
        with torch.no_grad():
            outputs = self.model.generate(
                **input_dict["input_text"],
                return_dict_in_generate=True,
                output_scores=True,
                #max_length=text_length+512,
                do_sample=True,
                #temperature = 0.7,
                top_k = self.args.top_k,
                top_p = self.args.top_p,
                repetition_penalty = self.args.rep_pen,
                eos_token_id = self.tokenizer.eos_token_id ,
                pad_token_id = 0,
                num_return_sequences = self.args.n_sample,
                max_new_tokens       = self.args.text_length,
            )
        # GreedySearchDecoderOnlyOutput(sequences=tensor([[seq1],[seq2]],device='cuda:0'), scores=None, attentions=None, hidden_states=None)

        answers = []
        
        for idx, sent in enumerate(outputs.sequences):
            result = self.tokenizer.decode(sent,skip_special_tokens=True)
            result = result.split(self.tokenizer.eos_token)[0]
            answer = result.split(sep=prompt,maxsplit=1)[1]
            #else:
            #    answer = result.split(sep="query:",maxsplit=1)[1]
            answers.append(answer)

        input_dict["answers"] = answers
        return input_dict        

    def metric(self,references ,candidates ,fn,n): # unigram bleu
        if fn == "bleu":
            scores = self.bleu_fn(references,candidates,n)
        elif fn == "f1":
            scores = self.f1_fn(references,candidates,n) 
        elif fn == 'distinct':
            scores = self.dist_fn(candidates)
        return scores

    def bleu_fn(references, candidates, n = 2):
        unigram, bigram = [],[]
        for ref, can in zip(references, candidates):
            #can = normalize_answer(can)
            reference = [" ".join(jieba.cut(ref)).split()]  # may have multiple ref, need [[ref1]]
            candidate = " ".join(jieba.cut(can)).split()

            chencherry = SmoothingFunction()

            score1, score2= sentence_bleu(reference,candidate,weights=[(1.0,),(0.5,0.5)],smoothing_function=chencherry.method7) #methods 7 ref prophetnet
            unigram.append(score1)
            bigram.append(score2)

        return sum(unigram) / len(unigram), sum(bigram) / len(bigram)

    def f1_fn(references, candidates, n = 2):
        def pre_recall_f1(reference,candidate):
            from collections import Counter
            common = Counter(reference) & Counter(candidate)
            num_same = sum(common.values())
            if num_same == 0:
                return 0, 0, 0
            precision = 1.0 * num_same / len(candidate)
            recall = 1.0 * num_same / len(reference)
            f1 = (2 * precision * recall) / (precision + recall)
            return precision, recall, f1

        pre, re, f1 = [],[],[]
        for ref, can in zip(references, candidates):
            #can = normalize_answer(can)
            if n == 1: # unigram
                reference = [[ref[i] for i in range(len(ref))]]
                candidate = [can[i] for i in range(len(can))]
            else: #token level
                reference = [" ".join(jieba.cut(ref)).split()]
                candidate = " ".join(jieba.cut(can)).split()
            
            (_pre, _re, _f1)  = [pre_recall_f1(r, candidate) for r in reference][0]
            pre.append(_pre)
            re.append(_re)
            f1.append(_f1)
        return sum(pre)/len(pre), sum(re)/len(re), sum(f1)/len(f1)

    # Ref: https://github.com/microsoft/ProphetNet/blob/master/GLGE_baselines/script/script/evaluate/personachat/eval.py
    def dist_fn(candidates):
        batch_size = len(candidates)
        intra_dist1, intra_dist2 = [],[]
        unigrams_all, bigrams_all = Counter(), Counter()

        for can in candidates:
            unigrams = Counter(can)
            bigrams = Counter(zip(can,can[1:]))
            intra_dist1.append((len(unigrams)+1e-12) / (len(can)+1e-5))
            intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(can)-1)+1e-5))

            unigrams_all.update(unigrams)
            bigrams_all.update(bigrams)

        inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
        inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
        intra_dist1 = sum(intra_dist1) / len(intra_dist1)
        intra_dist2 = sum(intra_dist2) / len(intra_dist2)
        return intra_dist1, intra_dist2, inter_dist1, inter_dist2
                

    # The four following functions are adapted from Meta ParlAI:
    # https://github.com/facebookresearch/ParlAI/blob/main/parlai/core/metrics.py
    @staticmethod
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        re_art = re.compile(r"\b(是|的|啊)\b")
        re_punc = re.compile(r"[!\"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\'，。？！]")
        def remove_articles(text):
            return re_art.sub(" ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            return re_punc.sub(" ", text)  # convert punctuation to spaces

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def evaluate(self):
        raise NotImplementedError
class DialogueTest(TestModule):
    def __init__(self, **args):
        super().__init__(**args)
        self.task = 'dial'

        data_file = f"/cognitive_comp/yangqi/data/DuSinc/dev_{self.task}.json"
        self.examples = self.load_from_json(data_file)

    def preprocess(self, item, context=1, max_kno_len=256, max_src_len=128, max_tgt_len=128):
        if "knowledge" in item:
            item["kno"] = item["knowledge"]  #convert name

        if "[SEP]" in item["src"]:
            src = " ".join(item["src"].split("[SEP]")[-1*context:]) #上下文处理不太好，所以只暴露最后几条上下文，相当于滑动 windows
        else:
            src = item["src"]
        src = self.truncate(src, max_src_len-2)
        kno = self.truncate(item["kno"],max_kno_len-2)
        tgt = self.truncate(item["tgt"], max_tgt_len-2)
        
        input_text = f'knowledge: {kno} context: {src} response:'
        input_text = self.tokenizer(input_text,return_tensors='pt')
        input_text.to(device)

        return {
            "input_text":input_text,
            "kno"       :kno,
            "src"       :src,
            "tgt"       :tgt
        }

    def generate(self, input_dict, prompt="response:"):
        return super().generate(input_dict, prompt)

    def evaluate(self, nums, metrics):
        nums = min(nums,len(self.examples))
        candidates, references = [],[]

        with st.spinner("正在评估中"):
            for idx in tqdm(range(nums)):
            #for idx in tqdm(range(100)):
                #kno, src, tgt = data[idx]
                item = self.examples[idx]
                input_dict = self.preprocess(item)
                output = self.generate(input_dict)

                candidates.append(output["answers"][0])
                references.append(output["tgt"])

            st.write("candidate")
            st.write(candidates)
            st.write("reference")
            st.write(references)

            if "bleu" in metrics:
                bleu_score = self.metric(references, candidates, fn="bleu" ,n=2)
                st.write(f"Bleu1/2 score on dev : {bleu_score[0]:.4f}/{bleu_score[1]:.4f}")
                st.write(f"Prec score on dev : {f1_score[0]:.4f}")
            
            if "f1" in metrics:
                f1_score = self.metric(references, candidates, fn="f1", n=1)
                st.write(f"Re   score on dev : {f1_score[1]:.4f}")
                st.write(f"F1   score on dev : {f1_score[2]:.4f}")

class QueryTest(TestModule):
    def __init__(self, **args):
        super().__init__(**args)
        self.task = 'dial'

        data_file = f"/cognitive_comp/yangqi/data/DuSinc/dev_{self.task}.json"
        self.examples = self.load_from_json(data_file)

    def preprocess(self, item, context=1, max_src_len=128, max_tgt_len=128):
        if "[SEP]" in item["src"]:
            src = " ".join(item["src"].split("[SEP]")[-1*context:]) #上下文处理不太好，所以只暴露最后几条上下文，相当于滑动 windows
        else:
            src = item["src"]
        src = self.truncate(src, max_src_len-2)
        kno = ""
        tgt = self.truncate(item["tgt"], max_tgt_len-2)
        
        input_text = f'knowledge: {kno} context: {src} response:'
        input_text = self.tokenizer(input_text,return_tensors='pt')
        input_text.to(device)

        return {
            "input_text":input_text,
            "src"       :src,
            "tgt"       :tgt
        }

    def generate(self, input_dict, prompt="query:"):
        return super().generate(input_dict, prompt)

 