import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import json
import argparse, sys
from tqdm import tqdm
import sys,time,os, random,re
sys.path.append("../../../")
from fengshen.utils import chinese_char_tokenize #内部分词
from fengshen.metric.eval_utils import load_metric_fn

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

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
        model_path = os.path.join(self.root_dir, model_path)
        if token_path is None:
            token_path = "/cognitive_comp/yangqi/model/Wenzhong-GPT2-110M"

        tokenizer = GPT2Tokenizer.from_pretrained(token_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)

        tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # [PAD]

        model.to(device)
        tokenizer.pad_token = '[PAD]'

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
    def truncate(document:str, max_num_tokens:int,reverse=True):
        total_length = len(document)
        if total_length <= max_num_tokens:
            return document
        else: 
            if reverse:
                return document[-1*max_num_tokens:]
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
                #max_length=self.args.text_length,
                do_sample=self.args.do_sample,
                num_beams=self.args.num_beams,
                temperature = self.args.temp,
                top_k = self.args.top_k,
                top_p = self.args.top_p,
                repetition_penalty = self.args.rep_pen,
                eos_token_id = self.tokenizer.eos_token_id ,
                pad_token_id = 0,
                num_return_sequences = self.args.n_sample,
                max_new_tokens = self.args.text_length - 256 - 128,
            )
        # GreedySearchDecoderOnlyOutput(sequences=tensor([[seq1],[seq2]],device='cuda:0'), scores=None, attentions=None, hidden_states=None)

        answers = []
        
        for idx, sent in enumerate(outputs.sequences):
            result = self.tokenizer.decode(sent,skip_special_tokens=True,clean_up_tokenization_spaces=True)
            result = result.split(self.tokenizer.eos_token)[0]
            answer = result.split(sep=prompt,maxsplit=1)[1]
            #else:
            #    answer = result.split(sep="query:",maxsplit=1)[1]
            answers.append(answer)

        input_dict["answers"] = answers
        return input_dict           

    # unified evaluate with transformer
    def fn(self,refs, cans, metric="bleu"):
        """
        Input: 单个 [ref] 和单个 can 列表, ref 内需要 [] 包裹多条 
            reference = [["哈哈，我也不会这些操作"],["是吗"]]
            candidate = ["哈哈哈 ，我只会说 。但是实际操作也不会","是的"]
            fn(reference, candidate)
        """
        fn = load_metric_fn(metric)
        if metric in ["dist","vocab"]:
            scores = fn(cans)
        elif metric in ["f1","bleu","em","rouge"]:
            scores = fn(refs, cans)
        else:
            print("Do not support other metrics")
        return scores    

    def evaluate(self):
        raise NotImplementedError

class DialogueTest(TestModule):
    def __init__(self, args):
        super().__init__(args)
        self.task = 'dial'
        self.data_file = self.args.data_file
        self.examples = self.load_dataset(self.data_file)
        self.runname = args.runname
        if not os.path.exists(self.root_dir + self.runname):
            os.makedirs(self.root_dir + self.runname)

        tm = time.localtime()
        self.log_file = self.root_dir + self.runname +f"/eval{tm.tm_mon}-{tm.tm_mday}-{tm.tm_hour}-{tm.tm_min}.json"

    def preprocess(self, item, context=1, max_kno_len=256, max_src_len=128, max_tgt_len=128, max_seq_len=512):
        if "knowledge" in item:
            item["kno"] = item["knowledge"]  #convert name

        if "[SEP]" in item["src"]:
            src = " ".join(item["src"].split("[SEP]")) #上下文处理不太好，所以只暴露最后几条上下文，相当于滑动 windows
        else:
            src = item["src"]
        src = self.truncate(src, max_src_len-2, reverse=True)
        kno = self.truncate(item["kno"], max_kno_len-2)
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

    def predict(self, output=True):
        tm = time.localtime()
        pfile = self.root_dir + self.runname + f"/predict{tm.tm_mon}-{tm.tm_mday}-{tm.tm_hour}.json"
        f = open(pfile,"w", encoding="utf-8")

        json.dump({"args": vars(self.args)},f,ensure_ascii=False)
        f.write("\n")
        
        # only predict adn output file
        nums = min(self.args.eg_num,len(self.examples))
        # nums = len(self.examples)
        candidates, references = [],[]
        
        print(f"Data Inference")
        for idx in tqdm(range(nums)):    
            item = self.examples[idx]
            input_dict = self.preprocess(item)
            output = self.generate(input_dict)

            candidates.append(output["answers"][0])
            references.append([output["tgt"]])

            json.dump({
                "kno": input_dict["kno"],
                "src": input_dict["src"],
                "can": output["answers"][0],
                "tgt": output["tgt"]
            },f,ensure_ascii=False)
            f.write("\n")
        f.close()
        return candidates, references

    def load_predict(self,load_file):
        cans, refs = [],[]
        with open(load_file,"r") as f:
            for line in f.readlines()[1:]:
                outputs = json.loads(line)
                can, ref = outputs["can"], outputs["tgt"]
                cans.append(can)
                refs.append([ref])
        return cans, refs

    def evaluate(self, metrics, load_file=None):
        f = open(self.log_file,'w+')

        if self.args.verbose:
            print(f"-----Begin Evaluate-------")
            f.write(f"Dataset file: {self.args.data_file} Length:{len(self.examples)}\n")
            f.write(f"------------ Args ------------")
            for key in list(vars(self.args).keys()):
                f.write(f"{key} : {vars(self.args)[key]}\n")

        if load_file:
            candidates, references = self.load_predict(self.root_dir + load_file)
        else:
            candidates, references = self.predict(output=True)
            print(candidates, references)

        scores = {}
        for m in metrics:
            print(f"Evaluate {m} score")
            assert len(references) == len(candidates)
            score = self.fn(references,candidates,m)
            scores.update(score)
 
        print(f"-----End Evaluate-------")
        f.write(f"------------ metrics ------------\n")
        for key in scores.keys():
            f.write(f"{key} score: {scores[key]}\n")
            print(f"{key} score: {scores[key]}\n")
        f.close()
        return scores

def parse_augment():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--runname',type=str)
    args_parser.add_argument('--dataset', type=str,default='duconv')
    args_parser.add_argument('--root_dir', type=str,default='/cognitive_comp/yangqi/model/')
    args_parser.add_argument('--gpu', type=bool,default=True)
    args_parser.add_argument('--data_file', type=str,default='/cognitive_comp/yangqi/data/DuSinc/dev_dial.json')
    args_parser.add_argument('--model_path', type=str,default='Wenzhong-Finetune-Loss0.1')
    args_parser.add_argument('--load_file', type=str,default=None)
    args_parser.add_argument('--token_path', type=str,default=None)
    args_parser.add_argument('--do_sample', type=bool,default=True)
    args_parser.add_argument('--n_sample', type=int,default=1)
    args_parser.add_argument('--text_length', type=int,default=512)
    args_parser.add_argument('--top_k', type=int,default=0)
    args_parser.add_argument('--top_p', type=float,default=0.9)
    args_parser.add_argument('--num_beams', type=int,default=4)
    args_parser.add_argument('--rep_pen', type=float,default=1.2)
    args_parser.add_argument('--temp', type=float,default=0.9)
    args_parser.add_argument('--context', type=int,default=1)
    args_parser.add_argument('--eg_num', type=int,default=10) #only test script
    args_parser.add_argument('--verbose', type=bool,default=True) 

    if len(sys.argv) == 1:
        args_parser.print_help()
        sys.exit(1)
    args = args_parser.parse_args()
    return args


device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parse_augment()
tester = DialogueTest(args)
# scores = tester.predict()
scores = tester.evaluate(metrics=["bleu","f1","dist","vocab","em","rouge"],load_file=args.load_file)
