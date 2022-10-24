from utils import test_dataloader, LabelSmoothingCrossEntropy
from utils import truncate_sequence, white_space_fix
from fengshen.utils import chinese_char_tokenize
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from fengshen.data.t5_dataloader.dialo_datasets import DialoT5DataModule
import time
import sys
import os
import json
import torch
import argparse
import pytorch_lightning as pl
from dataclasses import dataclass
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import BartForConditionalGeneration, BertTokenizer, AutoTokenizer, T5ForConditionalGeneration
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.functional import bleu_score, rouge_score
sys.path.append('../../../')


@dataclass
class QGT5Collator:
    @ staticmethod
    def add_data_specific_args(parent_args):
        # the hyperparameters should be determined according to the max length of context in dataset
        parser = parent_args.add_argument_group('BART DIalo Collator')
        parser.add_argument('--max_seq_length', default=512, type=int)
        parser.add_argument('--max_src_length', default=32, type=int)
        parser.add_argument('--max_kno_length', default=416, type=int)
        parser.add_argument('--max_tgt_length', default=64, type=int)
        parser.add_argument('--mask_ans_style', default='normal', type=str)
        return parent_args

    def __init__(self, tokenizer, args):
        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.print_example = True
        self.mask_ans_style = args.mask_ans_style
        self.do_eval_only = args.do_eval_only
        self.tokenizer_type = args.tokenizer_type

    def encode(self, x, y):
        # tokenize sentence
        if self.tokenizer_type == "bert":
            x = x
            y = y
        else:
            # t5 sentence piece
            x = self.tokenizer.bos_token + x + self.tokenizer.eos_token
            y = y + self.tokenizer.eos_token

        encoder_input = self.tokenizer.encode_plus(
            x,
            max_length=self.args.max_kno_length + self.args.max_src_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        decoder_output = self.tokenizer.encode_plus(
            y,
            max_length=self.args.max_tgt_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )

        return encoder_input, decoder_output

    def mask(self, s):
        def replace_span(source, target, sptoken):
            ans_bos, ans_eos = s["ans_span"][0]
            return source[:ans_bos] + sptoken + source[ans_eos:]

        def replace_all(source, target, sptoken):
            return source.replace(target, sptoken)

        if 'multispan' in self.mask_ans_style:
            fn = replace_all
        else:
            fn = replace_span

        # unmask: 北京是中国的首都
        if 'unmask' in self.mask_ans_style:
            return s["context"]

        # normal: 北京是 <mask> 的首都
        if 'normal' in self.mask_ans_style:
            self.anstoken = self.tokenizer.mask_token
            masked_context = fn(s["context"], s["answer"][0], self.anstoken)
            return masked_context

        # anstoken: 北京是 [ANS] 的首都
        if 'anstoken' in self.mask_ans_style:
            anstoken_dict = {
                "bert": "[ANS]",
                "bart": "<ans>"
            }
            self.anstoken = anstoken_dict[self.tokenizer_type]
            masked_context = fn(s["context"], s["answer"][0], self.anstoken)
            return masked_context

        # postag: 北京是 <beg> 中国 <eos> 的首都
        if 'postag' in self.mask_ans_style:
            begtoken, endtoken = "<beg>", "<eos>"
            self.anstoken = begtoken + s["answer"][0] + endtoken
            masked_context = fn(s["context"], s["answer"][0], self.anstoken)
            return masked_context

        return masked_context

    def prompt(self, context, answer, question):
        # pre_prompt, mid_prompt, post_prompt = "","</s>","" #prompt v1,v2
        # pre_prompt, mid_prompt, post_prompt = "knowledge:","answer:","" #prompt v5
        # pre_prompt, mid_prompt, post_prompt = "知识:","回答:","" #prompt v4
        pre_prompt, mid_prompt, post_prompt = "知识:", "回答:", "问题:"  # prompt v4

        context = truncate_sequence(context, self.args.max_kno_length-len(pre_prompt)-1)

        # used in squad-2.0
        # noted that src and tgt is reversed in qg
        answer = truncate_sequence(answer, self.args.max_src_length - len(mid_prompt)-1)
        question = truncate_sequence(question, self.args.max_tgt_length-len(post_prompt)-1)

        # x_trunc = f'{context} </s> {answer}'  # prompt v1
        # x_trunc = f' {answer} </s> {context}' #prompt v2
        x_trunc = f'{pre_prompt}{context}{mid_prompt}{answer}'  # prompt v3

        y_trunc = f'{post_prompt}{question}'
        return x_trunc, y_trunc

    def __call__(self, samples):
        """
        ans_num = 1 适用于 Train 数据只有 1 条 answer 取第一条情况
        ans_num > 1 适用于 Dev 数据有多条 answer 情况
        Input:
        input_ids: input_ids (text + answer)
        attn_mask: input attn mask
        labels:   decoder_ids (question)
        """
        input_ids, attn_mask, labels, decoder_attn_mask = [], [], [], []
        ans, qes, ctx, ans_spans, idxs, imp = [], [], [], [], [], []

        for s in samples:
            if self.do_eval_only:
                # log origin answer
                ans.append(s["answer"])
                qes.append(s["question"])
                ctx.append(s["context"])
                ans_spans.append(s["ans_span"])
                idxs.append(s["idx"])

            if "is_impossible" in s:
                imp.append(s["is_impossible"])
            else:
                imp.append(False)  # SQUAD 1.0 don't have is_impossible

            if not s["is_impossible"]:  # have ans and ans_span
                context = self.mask(s)
                answer = s["answer"][0]
                question = s["question"]
            else:  # no ans and ans_span
                context = s["context"]
                answer = "无答案"
                question = s["question"]

            x_trunc, y_trunc = self.prompt(context, answer, question)
            print("x_trunc y_trunc after prompt")
            print(x_trunc, y_trunc)
            encoder_input, decoder_output = self.encode(x_trunc, y_trunc)

            input_ids.append(encoder_input["input_ids"])
            attn_mask.append(encoder_input["attention_mask"])
            labels.append(decoder_output["input_ids"])

        labels = torch.cat(labels)
        print(labels.shape)
        if self.tokenizer_type == "bart":
            end_token_index = torch.where(labels == self.tokenizer.eos_token_id)[1]
        else:
            end_token_index = torch.where(labels == self.tokenizer.sep_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx + 1:] = -100  # cross entropy cal

        data = {
            'input_ids': torch.cat(input_ids),
            'attention_mask': torch.cat(attn_mask),
            'labels': labels
        }
        if self.do_eval_only:
            data.update({
                'answer': ans,
                'question': qes,
                'context': ctx,
                'ans_span': ans_spans,
                'idx': idxs,
                'is_impossible': imp
            })

        if self.print_example:
            print(x_trunc)
            print(y_trunc)
            self.print_example = False

        import pickle
        f = open("collator_output.pkl", "wb")
        pickle.dump({
            "data": data,
            "end_token": end_token_index
        }, f)
        return data


class BARTFinetuneModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--model_path', type=str, default='')
        parser.add_argument('--learning_rate', default=1e-5, type=float)  # config optimizer
        parser.add_argument('--weight_decay', default=0.1, type=float)
        parser.add_argument('--warmup', default=0.01, type=float)
        parser.add_argument('--label_smooth', default=0, type=float)
        parser.add_argument('--new_token_path', default="./", type=str)
        #parser.add_argument('--keep_tokens_path', default=None, type=str)
        return parent_args

    def __init__(self, tokenizer, args):
        super().__init__()
        self.save_hyperparameters(args)
        if 'BART' in args.model_path:
            self.model = BartForConditionalGeneration.from_pretrained(args.model_path)
        elif 'T5' in args.model_path:
            self.model = T5ForConditionalGeneration.from_pretrained(args.model_path)
        self.tokenizer = tokenizer

        # add special token ans
        # self.tokenizer.save_vocabulary(self.args.model_path)
        new_vocab = args.model_path+"/sp_vocab/"
        if not os.path.exists(new_vocab):
            os.makedirs(new_vocab)
        self.tokenizer.save_pretrained(new_vocab)
        self.model.resize_token_embeddings(len(tokenizer))
        self.vocab_size = len(tokenizer)
        self.rougescore = ROUGEScore(rouge_keys=('rougeL'), normalizer=lambda x: x)

        if self.hparams.label_smooth:
            self.loss_fct = LabelSmoothingCrossEntropy(smoothing=0.1)

    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()

            # Calculate total steps
            if self.trainer.max_epochs > 0:
                world_size = self.trainer.world_size
                tb_size = self.hparams.train_batchsize * max(1, world_size)
                ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
                self.total_steps = (len(train_loader.dataset) *
                                    self.trainer.max_epochs // tb_size) // ab_size
            else:
                self.total_steps = self.trainer.max_steps // self.trainer.accumulate_grad_batches

            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        from fengshen.models.model_utils import configure_optimizers
        return configure_optimizers(self)

    def training_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])

        loss = output.loss
        if self.hparams.label_smooth:
            loss = self.loss_fct(output.logits.view(-1, self.vocab_size), batch["labels"].view(-1))

        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])
        acc = self.compute_acc(output.logits, batch['labels'])
        self.log('val_loss', output.loss, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)
        self.log('val_ppl', torch.exp(output.loss), sync_dist=True)

        cond_output = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            do_sample=True,
            num_beams=5,
            early_stopping=True,
            max_length=64
            top_p=0.9,
        )

        batch_label = torch.where(batch["labels"] != -100, batch["labels"], self.tokenizer.pad_token_id)
        pred = self.tokenizer.batch_decode(cond_output, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        ques = self.tokenizer.batch_decode(batch_label, clean_up_tokenization_spaces=True, skip_special_tokens=True)

        pred = [chinese_char_tokenize(white_space_fix(p)) for p in pred]
        ques = [chinese_char_tokenize(white_space_fix(l)) for l in ques]
        self.rougescore.update(pred, ques)

        return pred

    def validation_epoch_end(self, validation_step_outputs):
        rouge = self.rougescore.compute()
        self.log('val_rouge', rouge["rougeL_fmeasure"], sync_dist=True)

    def on_predict_start(self):
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    def predict_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'])

        loss_tensor = self.loss_fct(output.logits.transpose(1, 2), batch["labels"])
        if self.hparams.tokenizer_type == 'bart':
            eos_index = torch.where(batch['labels'] == self.tokenizer.eos_token_id)[1]
        elif self.hparams.tokenizer_type == 'bert':
            eos_index = torch.where(batch['labels'] == self.tokenizer.sep_token_id)[1]

        loss = torch.sum(loss_tensor, dim=1) / eos_index
        print(loss.shape)

        with torch.no_grad():
            cond_output = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                do_sample=True,
                num_beams=5,
                max_length=64,
                top_p=0.9,
                output_scores=True,
                return_dict_in_generate=True
            )

        pred = self.tokenizer.batch_decode(
            cond_output.sequences, clean_up_tokenization_spaces=True, skip_special_tokens=True)  # ['sequences']
        pred = [white_space_fix(p) for p in pred]  # remove prompt and white space
        score = cond_output.sequences_scores
        return pred, score, loss

    def compute_acc(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float())/y_true.shape[0]
        return acc

    def compute_bleu(self, candidates, references):
        references = [chinese_char_tokenize(ref[0]) for ref in references]
        candidates = [chinese_char_tokenize(can) for can in candidates]
        bleus = {}
        for i in range(1, 5):
            bleus['bleu_'+str(i)] = bleu_score(candidates, references, i, False)
        return bleus

    def compute_rouge(self, candidates, references):
        references = [chinese_char_tokenize(ref) for ref in references]
        candidates = [chinese_char_tokenize(can) for can in candidates]

        def normalize(x):
            return x
        rouges = rouge_score(candidates, references, normalizer=normalize)
        return rouges

    def compute_f1(self, candidates, references):
        from fengshen.metric.eval_utils import f1_fn

        f1 = f1_fn(references, candidates)
        return f1

    def on_save_checkpoint(self, checkpoint) -> None:
        # Save the current loop info in the mid of epoch
        # if you lightning <= 1.6.0  uncomment the line below
        # checkpoint['loops'] = self.trainer.checkpoint_connector._get_loops_state_dict()
        if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
            self.model.save_pretrained(os.path.join(
                self.trainer.checkpoint_callback.dirpath,
                'hf_pretrained_epoch{}_step{}'.format(checkpoint['epoch'], checkpoint['global_step'])))

    def on_load_checkpoint(self, checkpoint) -> None:
        global_step_offset = checkpoint["global_step"]
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset


def get_time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def get_tokenizer(tokenizer_type, pretrained_model_path):
    if tokenizer_type == 'bart':
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path, use_fast=False, additional_special_tokens=["<ans>", "<beg>", "<end>"])
        print(len(tokenizer))
    elif tokenizer_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_path, use_fast=False, additional_special_tokens=["[ANS]"])
    return tokenizer


def load_predict(load_file):
    """Load from etst_pred.json"""
    with open(load_file, "r") as f:
        res = f.readlines()
        data = json.loads(res[0])
    return data['pred']


def main():
    total_parser = argparse.ArgumentParser("Finetune BART for QG")
    total_parser.add_argument('--do_eval_only', action='store_true', default=False)
    total_parser.add_argument('--tokenizer_type', type=str, default="bart")
    total_parser.add_argument('--tensorboard_dir', type=str, default="bart")
    total_parser.add_argument('--test_model', type=str, default="last.ckpt")
    total_parser.add_argument('--test_file', type=str, default="test.json")
    total_parser.add_argument('--sample_num', type=int, default=0)
    total_parser.add_argument('--pred_file', type=str, default="pred.json")
    total_parser.add_argument('--qa_type', type=int, default=2)
    total_parser.add_argument('--collator', type=str, default=None)
    total_parser.add_argument('--deepspeed')

    # Args for data preprocessing
    total_parser = DialoT5DataModule.add_data_specific_args(total_parser)
    total_parser = QGT5Collator.add_data_specific_args(total_parser)

    # Args for training
    total_parser = Trainer.add_argparse_args(total_parser)
    total_parser = UniversalCheckpoint.add_argparse_args(total_parser)
    total_parser = BARTFinetuneModel.add_model_specific_args(total_parser)

    # Args for base model
    args = total_parser.parse_args()

    tokenizer = get_tokenizer(args.tokenizer_type, args.model_path)
    collator = QGT5Collator(tokenizer=tokenizer, args=args)

    data_model = DialoT5DataModule(collate_fn=collator, tokenizer=tokenizer, args=args)

    if args.deepspeed is not None:
        os.environ['PL_DEEPSPEED_CONFIG_PATH'] = args.deepspeed

    model = BARTFinetuneModel(tokenizer, args)
    checkpoint_callback = UniversalCheckpoint(args).callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = loggers.TensorBoardLogger(save_dir=args.tensorboard_dir, name="tf_log")
    trainer = Trainer.from_argparse_args(args,
                                         logger=logger,
                                         callbacks=[checkpoint_callback, lr_monitor]
                                         )
    if not args.do_eval_only:
        trainer.fit(model, data_model)
    else:
        test_data = test_dataloader(collate_fn=collator, args=args)
        preds_batches = trainer.predict(model, test_data, ckpt_path=args.dirpath+args.test_model)

        newpred_dir = os.path.join(args.default_root_dir, "test")
        if not os.path.exists(newpred_dir):
            os.makedirs(newpred_dir)
        newpred_file = os.path.join(newpred_dir, args.test_model + '_' + args.pred_file)
        f = open(newpred_file, 'w+')

        for batch_idx, data in enumerate(test_data):
            ans_batch, qes_batch, ctx_batch, span_batch, idx_batch, imp_batch = data["answer"], data[
                "question"], data["context"], data["ans_span"], data["idx"], data["is_impossible"]

            pred_batch = preds_batches[batch_idx]

            for idx in range(args.test_batchsize):
                output = {
                    "context": ctx_batch[idx],
                    "answer": ans_batch[idx],
                    "ans_span": span_batch[idx],
                    "question": pred_batch[idx],
                    "idx": idx_batch[idx],
                    "is_impossible": False,
                    "qa_type": args.qa_type,
                }

                pred, score, loss = preds_batches[batch_idx]
                output["prediction"] = pred[idx]
                output["is_impossible"] = imp_batch[idx]
                output["score"] = float(score[idx])
                output["loss"] = float(loss[idx])

                json.dump(output, f, ensure_ascii=False)
                f.write('\n')


if __name__ == '__main__':
    main()
