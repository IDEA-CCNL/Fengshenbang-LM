

import torch
from tqdm import tqdm
# predictor
from uie.seq2seq.t5_bert_tokenizer import T5BertTokenizer
from transformers import AutoModelForSeq2SeqLM
from uie.extraction.scorer import *
from uie.sel2record.sel2record import SEL2Record

#parser
import re
from nltk.tree import ParentedTree
from uie.extraction.constants import type_end, type_start


class UIEPredictor:
    # @staticmethod
    # def add_data_specific_args(parent_args):
    #     return parent_args
    
    def __init__(self, model_path, schema, max_source_length=256,
                 max_target_length=192) -> None:
        
        self.tokenizer = T5BertTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.schema = schema
        self.model.cuda()
        self.model.eval() # !!
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.special_to_remove = {'<pad>', '</s>'}

    def schema_to_ssi(self):
        # Convert Schema to SSI
        # <spot> spot type ... <asoc> asoc type <text>
        ssi = "<spot>" + "<spot>".join(sorted(self.schema['type_list']))
        ssi += "<asoc>" + "<asoc>".join(sorted(self.schema['role_list']))
        ssi += "<extra_id_2>"
        return ssi

    def show_ssi(self):
        ssi = "<spot>" + "<spot>".join(sorted(self.schema['type_list']))
        ssi += "<asoc>" + "<asoc>".join(sorted(self.schema['role_list']))
        ssi += "<extra_id_2>"
        print(ssi)

    def post_processing(self, x):
        for special in self.special_to_remove:
            x = x.replace(special, '')
        return x.strip()

    @torch.no_grad()
    def predict_batch_constriant(self, text):
        ssi = self.schema_to_ssi()
        text = [ssi + x for x in text] 
        force_word = ['<extrad_id_5>'+ x for x in text] 
        force_words_ids = self.tokenizer(force_word, padding=True, add_special_tokens=False)['input_ids']
        inputs = self.tokenizer(text, padding=True, return_tensors='pt').to(self.model.device)

        inputs['input_ids'] = inputs['input_ids'][:, :self.max_source_length]
        inputs['attention_mask'] = inputs['attention_mask'][:, :self.max_source_length]
        pred = self.model.generate(
            input_ids=inputs['input_ids'],
            num_beams=5,
            force_words_ids=force_words_ids,
            attention_mask=inputs['attention_mask'],
            max_length=self.max_target_length, )

        pred = self.tokenizer.batch_decode(pred, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        return [self.post_processing(x) for x in pred]


    @torch.no_grad()
    def predict_batch(self, text):
        ssi = self.schema_to_ssi()
        text = [ssi + x for x in text] 
        inputs = self.tokenizer(text, padding=True, return_tensors='pt').to(self.model.device)

        inputs['input_ids'] = inputs['input_ids'][:, :self.max_source_length]
        inputs['attention_mask'] = inputs['attention_mask'][:, :self.max_source_length]

        pred = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=self.max_target_length, )

        pred = self.tokenizer.batch_decode(pred, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        return [self.post_processing(x) for x in pred]


class UIEParser:
    def __init__(self) -> None:
        self.split_bracket = re.compile(r"\s*<extra_id_\d>\s*")
        self.left_bracket = '【'
        self.right_bracket = '】'
    
    def add_space(self, text):
        """
            add space between special token
        """
        new_text_list = list()
        for item in zip(self.split_bracket.findall(text), self.split_bracket.split(text)[1:]):
            new_text_list += item
        return ' '.join(new_text_list)
    

    def convert_bracket(self, text): 
        """
            <extra_id_0><extra_id_1>' -> "【 】“
        """
        text = self.add_space(text) 
        for start in [type_start]:
            text = text.replace(start, self.left_bracket)
        for end in [type_end]:
            text = text.replace(end, self.right_bracket)
        return text

    def show(self, pred):
        for p in pred:
            print(self.parser(p).pretty_print())

    def parser(self, p):
        p = self.convert_bracket(p)
        p_tree = ParentedTree.fromstring(p, brackets=self.left_bracket+ self.right_bracket)
        return p_tree

    def parser_batch(self, pred):
        p_trees = []
        for p in pred:
            p = self.convert_bracket(p)
            p_tree = ParentedTree.fromstring(p, brackets=self.left_bracket+ self.right_bracket)
            p_trees.append(p_tree)
        return p_tree
            

def main():
    schema ={
        'type_list': [ "自然地点", "地理区域", "人物"],
        'role_list': [],
        'type_role_dict':{"人名": [], "自然地点": [], "地理区域": []}
    }

    model_path='/cognitive_comp/yangjing/UIE/FengshenUIE/Fengshen-UIE-NER'

    predictor = UIEPredictor(model_path=model_path, schema=schema, 
                 max_source_length=256, max_target_length=128)
    predictor.show_ssi()

    text = ["我叫赵新民是不是很说话？南京市长江大桥位于南京市政府东北方向"]
    pred = predictor.predict_batch(text)

    parser = UIEParser()
    parser.show(pred)
     

    
if __name__ == '__main__':
    main()




