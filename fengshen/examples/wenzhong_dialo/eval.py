import argparse
from collections import Counter
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os, random, re
import torch
import json
import jieba
import argparse, sys
from tqdm import tqdm
from torchmetrics.functional import bleu_score
#from torchtext.data.metrics import bleu_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestModule:
    def __init__(self, args):
        #self.args = argparse.Namespace(**args) # if use args
        self.args = args
        self.model_path = args.model_path
        self.root_dir = self.args.root_dir
        self.tokenizer, self.model = self.load_model(args.model_path)

    @staticmethod
    def set_seed(seed=42):
        random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def load_model(self, model_path, token_path=None):
        # if "epoch" in model_id:
        #     epoch = model_id.split('epoch')[-1]
        #     step = (int(epoch) + 1)*10000
        #     model_path = f"{rootdir}hf_pretrained_epoch{epoch}_step{step}"
        # else:
        #     model_path = "/cognitive_comp/yangqi/model/Wenzhong-GPT2-110M" 
        model_path = os.path.join(self.root_dir, model_path)
        if token_path is None:
            token_path = "/cognitive_comp/yangqi/model/Wenzhong-GPT2-110M"

        tokenizer = GPT2Tokenizer.from_pretrained(token_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)

        model.to(device)
        if self.args.verbose:
            print(f"Load model from {model_path} and tokenizer from {tokenizer}")
        return tokenizer, model
    
    def load_dataset(self, data_file):
        # from json
        examples = []
        with open(data_file,'r',encoding='utf8')as fp:
            while True:
                line = fp.readline()
                if not line: #EOF
                    break
                s = json.loads(line)
                examples.append(s)
        if self.args.verbose:
            print(f"Load test file from {data_file} Length {len(examples)}")
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
                max_length=self.args.text_length,
                do_sample=True,
                num_beams=self.args.num_beams,
                temperature = self.args.temp,
                top_k = self.args.top_k,
                top_p = self.args.top_p,
                repetition_penalty = self.args.rep_pen,
                eos_token_id = self.tokenizer.eos_token_id ,
                pad_token_id = 0,
                num_return_sequences = self.args.n_sample
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

    @staticmethod
    def bleu_fn(references, candidates, n=2):
        unigram, bigram = [],[]
        for ref, can in zip(references, candidates):
            reference = [[ref]]
            candidate = [can]
            #reference = [[ref[i] for i in range(len(ref))]]
            #candidate = [can[i] for i in range(len(can))]
            #can = normalize_answer(can)
            #reference = [" ".join(jieba.cut(ref)).split()]  # may have multiple ref, need [[ref1]]
            #candidate = " ".join(jieba.cut(can)).split()

            #chencherry = SmoothingFunction()
            #score1, score2= sentence_bleu(reference,candidate,weights=[(1.0,),(0.5,0.5)]) #methods 7 ref prophetnet
            score1 = bleu_score(candidate,reference,1,False)
            score2 = bleu_score(candidate,reference,2,False)
            unigram.append(score1)
            bigram.append(score2)

        return sum(unigram) / len(unigram), sum(bigram) / len(bigram)

    @staticmethod
    def f1_fn(references, candidates, n=1):
        def pre_recall_f1(reference,candidate):
            from collections import Counter
            common = Counter(reference) & Counter(candidate)
            num_same = sum(common.values()) #TP 
            if num_same == 0:
                return 0, 0, 0
            precision = 1.0 * num_same / len(candidate)
            recall = 1.0 * num_same / len(reference)
            f1 = (2 * precision * recall) / (precision + recall)
            return precision, recall, f1

        pre, re, f1 = [],[],[]
        for ref, can in zip(references, candidates):
            #can = normalize_answer(can)
            if n == 1: # token live
                reference = [[ref[i] for i in range(len(ref))]]
                candidate = [can[i] for i in range(len(can))]
            else: # word level
                reference = [" ".join(jieba.cut(ref)).split()]
                candidate = " ".join(jieba.cut(can)).split()
            
            (_pre, _re, _f1)  = [pre_recall_f1(r, candidate) for r in reference][0]
            pre.append(_pre)
            re.append(_re)
            f1.append(_f1)
        return sum(pre)/len(pre), sum(re)/len(re), sum(f1)/len(f1)

    def acc_fn(references, candidates):
        num_same = 0 # 2-classification
        for ref, can in zip(references, candidates):
            ref = 0 if ref == "不检索" else 1
            can = 0 if can == "不检索" or "不" in can else 1
            if ref == can:
                num_same = num_same + 1
        
        return num_same / len(candidates)            

    # Ref: https://github.com/microsoft/ProphetNet/blob/master/GLGE_baselines/script/script/evaluate/personachat/eval.py
    @staticmethod
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

        inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5) #所有回复中构成的词典 unigram 长度/长度
        inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
        intra_dist1 = sum(intra_dist1) / len(intra_dist1) #每一句计算 dist 求均值 
        intra_dist2 = sum(intra_dist2) / len(intra_dist2)
        return inter_dist1, inter_dist2, intra_dist1, intra_dist2 

    def ppl_fn(reference,candidates):
        pass

    def rouge_fn(reference,candidates):
        pass

    # direct get vocab length
    def vocab_fn(candidates):
        pass
        

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
    def __init__(self, args):
        super().__init__(args)
        self.task = 'dial'

        data_file = self.args.data_file
        #data_file = f"/cognitive_comp/yangqi/data/DuSinc/dev_{self.task}.json"
        #data_file = f"/cognitive_comp/yangqi/data/DuSinc/test_{self.task}_a.json"
        
        self.examples = self.load_dataset(data_file)

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

    def evaluate(self, metrics):
        if self.args.verbose:
            print(f"-----Begin Evaluate-------")
        nums = min(self.args.eg_num,len(self.examples))
        candidates, references = [],[]

        for idx in tqdm(range(nums)):
        #for idx in tqdm(range(100)):
            #kno, src, tgt = data[idx]
            item = self.examples[idx]
            input_dict = self.preprocess(item)
            output = self.generate(input_dict)

            candidates.append(output["answers"][0])
            references.append(output["tgt"])

        scores = {}
        if "bleu" in metrics:
            scores["bleu1"], scores["bleu2"] = self.bleu_fn(references, candidates, 2)

        if "f1" in metrics:
            scores["pre"],scores["re"], scores["f1"] = self.f1_fn(references, candidates, 1)

        if "dist" in metrics:
            scores["inter_dist1"],scores["inter_dist2"],scores["intra_dist1"],scores["intra_dist2"] = self.dist_fn(candidates)

        if self.args.verbose:
            print(f"-----End Evaluate-------")
        self.info(candidates,scores)
        return scores

    def info(self, candidates, scores):
        f = open("./eval_test.txt",'w')
        f.write(f"Dataset file: {self.args.data_file} Length:{len(self.examples)}")

        f.write(f"------------ metrics ------------")
        for key in scores.keys():
            f.write(f"{key} score: {scores[key]}\n")
        
        f.write(f"------------ answers ------------")
        for can in candidates:
            f.write(can+'\n')


class QueryTest(TestModule):
    def __init__(self, **args):
        super().__init__(**args)
        self.task = 'dial'

        data_file = f"/cognitive_comp/yangqi/data/DuSinc/dev_{self.task}.json"
        self.examples = self.load_dataset(data_file)

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

    def evaluate(self):
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

            scores = {}
            if "bleu" in metrics:
                bleu_score = self.bleu_fn(references, candidates, 2)
                scores["bleu"] = bleu_score

            if "f1" in metrics:
                f1_score = self.f1_fn(references, candidates, 1)
                scores["f1"] = f1_score

            if "dist" in metrics:
                dist_score = self.dist_fn(candidates)
                scores["dist"] = dist_score

        return scores     


def parse_augment():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--runname',type=str)
    args_parser.add_argument('--dataset', type=str,default='duconv')
    args_parser.add_argument('--root_dir', type=str,default='/cognitive_comp/yangqi/model/')
    args_parser.add_argument('--data_file', type=str,default='/cognitive_comp/yangqi/data/DuSinc/dev_dial.json')
    args_parser.add_argument('--model_path', type=str,default='Wenzhong-Finetune-Loss0.1')
    args_parser.add_argument('--token_path', type=str,default=None)
    args_parser.add_argument('--n_sample', type=int,default=1)
    args_parser.add_argument('--text_length', type=int,default=512)
    args_parser.add_argument('--top_k', type=int,default=0)
    args_parser.add_argument('--top_p', type=float,default=0.6)
    args_parser.add_argument('--num_beams', type=int,default=4)
    args_parser.add_argument('--rep_pen', type=float,default=1.6)
    args_parser.add_argument('--temp', type=float,default=1.2)
    args_parser.add_argument('--context', type=int,default=1)
    args_parser.add_argument('--eg_num', type=int,default=10) #only test script
    args_parser.add_argument('--verbose', type=bool,default=True) 

    if len(sys.argv) == 1:
        args_parser.print_help()
        sys.exit(1)
    args = args_parser.parse_args()
    return args

args = parse_augment()
tester = DialogueTest(args)
scores = tester.evaluate(["bleu","f1","dist"])