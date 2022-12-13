from transformers import GPT2LMHeadModel, AutoTokenizer
from pytorch_lightning.callbacks import LearningRateMonitor
from torchmetrics.text.rouge import ROUGEScore
from dataclasses import dataclass
from fengshen.models.model_utils import configure_optimizers
from fengshen.utils import chinese_char_tokenize
from fengshen.examples.finetune_bart_qg.utils import LabelSmoothingCrossEntropy, truncate_sequence, white_space_fix
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.utils import UniversalCheckpoint
from pytorch_lightning import LightningModule, Trainer, loggers
import os
import argparse
import torch
import logging
import traceback
import json
import sys
import re
sys.path.append('../../../')


def findall(all_str, find_str):
    return [substr.start() for substr in re.finditer(find_str, all_str)]


def clean(x):
    prompt = "bot:"
    index = findall(x, prompt)
    if len(index) == 0:
        return x
    return x[index[-1]+len(prompt):]


@dataclass
class Collator:
    @staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Wenzhong Text Filling Collator')
        parser.add_argument('--max_seq_length', default=512, type=int)
        parser.add_argument('--max_src_length', default=256, type=int)
        parser.add_argument('--max_kno_length', default=128, type=int)
        parser.add_argument('--max_tgt_length', default=128, type=int)
        return parent_args

    def __init__(self, tokenizer, args, do_eval=False):
        if args.collator == "dial":
            self.collator = DialoCollator(tokenizer, args, do_eval)
        elif args.collator == "nokno_dial":
            self.collator = NoKnoDialoCollator(tokenizer, args)
        else:
            return NotImplementedError

    def __call__(self, samples):
        return self.collator.__call__(samples)

# Data example
# {"kno": "第一步：贴假睫毛前，先把睫毛卷起来，以免造成真假睫毛脱层。剪的时候分为三段，从睫毛根部开始，剪中间和末端。第二步：夹紧睫毛后涂一层睫毛膏，使睫毛定型，保持卷曲不下垂。第三步：戴假睫毛前先修剪一下。建议把假睫毛剪下来，分成三段，如图。选择中间部分使用。第四步：对假睫毛进行柔软的身体锻炼，握住假睫毛的两端，前后弯曲，
# 使假睫毛更贴合你的眼睛。第五步：将胶水涂在假睫毛的茎上，胶水半干时，从眼尾到眼中部贴在真睫毛上。第六步：再次睫毛膏，加强头部睫毛的密度，让真睫毛和假睫毛融合在一起。第七步：用眼线填充睫毛根部的缝隙，掩盖胶水的痕迹。第八步：扬起睫毛膏，下睫毛，让下睫毛也有存在感。第九步：贴完假睫毛后，别忘了用电动睫毛棒从根部梳理，让真假睫毛的曲度一致。第十步：之后，用眼线笔填充眼睛末端的线条，使眼线笔线条更加流畅。（来源：视觉中国）",
# "ctx": ["我不会贴假睫毛呀，好难！"], "tgt": "这个我专门了解过的。先把真睫毛涂一层睫毛膏卷起来定型，然后把假睫毛剪下来揉软，再把睫毛贴上胶水贴到眼皮上。", "idx": 0, "topic": ["购物", "化妆品", ""], "loc": "安徽省芜湖市", "query": "怎么贴假睫毛"}


@dataclass
class DialoCollator:
    tokenizer: None
    max_seq_length: int = 512
    max_kno_length: int = 256
    max_src_length: int = 128
    max_tgt_length: int = 128

    @ staticmethod
    def add_data_specific_args(parent_args):
        parser = parent_args.add_argument_group('Wenzhong Text Filling Collator')
        parser.add_argument('--max_seq_length', default=512, type=int)
        parser.add_argument('--max_src_length', default=128, type=int)
        parser.add_argument('--max_kno_length', default=256, type=int)
        parser.add_argument('--max_tgt_length', default=128, type=int)
        return parent_args

    def __init__(self, tokenizer, args, do_eval=False):
        self.tokenizer = tokenizer
        self.args = args
        self.max_seq_length = args.max_seq_length
        self.do_eval = do_eval

    def set_pad_100(self, x):
        x1 = x.clone().detach()
        x1[x == self.tokenizer.pad_token_id] = -100
        return x1

    def encode(self, x):
        # x = self.tokenizer.bos_token + x + self.tokenizer.eos_token
        input_dicts = self.tokenizer.encode_plus(
            x,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )

        # embedding
        input_ids = input_dicts["input_ids"]
        attn_mask = input_dicts["attention_mask"]
        labels = self.set_pad_100(input_ids)

        return [input_ids, labels, attn_mask]

    def __call__(self, samples, test=False):
        sep_token = self.tokenizer.sep_token
        kno_prompt, ctx_prompt, res_prompt, user_prompt = "knowledge:", "context:", "bot:", "user:"  # prompt
        tmp = []
        input_ids, labels, attn_mask = [], [], []
        kno, src, tgt = [], [], []
        for s in samples:
            if len(s["kno"]) != 0:
                knowledge = truncate_sequence(kno_prompt+s["kno"], self.args.max_kno_length-1)
                max_ctx_length = self.args.max_src_length
            else:
                knowledge = ""
                max_ctx_length = self.args.max_src_length+self.args.max_kno_length

            # worser
            # for idx,c in enumerate(s["ctx"][:-1]):
                # tmp.append(res_prompt + c if idx % 2 == 1 else user_prompt + c)

            context = ctx_prompt + truncate_sequence(sep_token.join(tmp), max_ctx_length-1-2-32, reverse=True)
            source = truncate_sequence(user_prompt+s["ctx"][-1], 32-1)
            target = truncate_sequence(res_prompt+s["tgt"], self.args.max_tgt_length)

            if self.args.do_eval_only or self.do_eval:
                x = f'{knowledge}{sep_token}{context}{sep_token}{source}{sep_token}{res_prompt}'
            else:
                x = f'{knowledge}{sep_token}{context}{sep_token}{source}{sep_token}{target}'
            # print("line113")
            print(x)
            g = self.encode(x)
            # print("line116")
            # print(g)

            input_ids.append(g[0])
            labels.append(g[1])
            attn_mask.append(g[2])
            kno.append(s["kno"])
            src.append(sep_token.join(s["ctx"]))
            tgt.append(s["tgt"])

        output = {
            'input_ids': torch.cat(input_ids),
            'attention_mask': torch.cat(attn_mask),
            'labels': torch.cat(labels),
            "target": tgt
        }
        if self.args.do_eval_only:
            output.update({
                "knowledge": kno,
                "source": src,
                "target": tgt
            })

        return output


@dataclass
class NoKnoDialoCollator(DialoCollator):
    """{"context": ["喂，吉姆，晚饭后去喝几杯啤酒怎么样？"], "response": "你知道这很诱人，但对我们的身体不好。", "dataname": "dailydialog", "id": 0}"""

    def __init__(self, tokenizer, args):
        super(NoKnoDialoCollator, self).__init__(tokenizer, args)

    def __call__(self, samples, test=False):
        sep_token = self.tokenizer.sep_token
        ctx_prompt, res_prompt, user_prompt = "context:", "bot:", "user:"  # prompt
        tmp = []
        input_ids, labels, attn_mask = [], [], []
        src, tgt = [], []
        for s in samples:
            context = ctx_prompt + truncate_sequence(sep_token.join(tmp), self.args.max_src_length-1-2-32, reverse=True)
            source = truncate_sequence(user_prompt+s["context"][-1], 32-1)
            target = truncate_sequence(res_prompt+s["response"], self.args.max_tgt_length)

            if self.args.do_eval_only or test:
                x = f'{context}{sep_token}{source}{sep_token}{res_prompt}'
            else:
                x = f'{context}{sep_token}{source}{sep_token}{target}'
            # print("line113")
            print(x)
            g = self.encode(x)
            # print("line116")
            # print(g)

            input_ids.append(g[0])
            labels.append(g[1])
            attn_mask.append(g[2])
            src.append(sep_token.join(s["context"]))
            tgt.append(s["response"])

        output = {
            'input_ids': torch.cat(input_ids),
            'attention_mask': torch.cat(attn_mask),
            'labels': torch.cat(labels),
            "target": tgt
        }
        if self.args.do_eval_only:
            output.update({
                "source": src,
                "target": tgt
            })

        return output


class GPT2Finetuner(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--model_path', type=str, default='')
        parser.add_argument('--learning_rate', default=1e-5, type=float)
        parser.add_argument('--min_learning_rate', default=1e-7, type=float)
        parser.add_argument('--lr_decay_steps', default=0, type=int)
        parser.add_argument('--lr_decay_ratio', default=1.0, type=float)
        parser.add_argument('--weight_decay', default=0.1, type=float)
        parser.add_argument('--warmup_steps', default=1000, type=int)
        parser.add_argument('--warmup_ratio', default=0.01, type=float)
        parser.add_argument('--label_smooth', default=0, type=float)
        parser.add_argument('--new_token_path', default="./", type=str)  # save new token after add special token
        parser.add_argument('--adam_beta1', default=0.9, type=float)
        parser.add_argument('--adam_beta2', default=0.999, type=float)
        parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        parser.add_argument('--scheduler_type', default='polynomial', type=str)

        return parent_args

    def __init__(self, tokenizer, collator, args) -> None:
        super().__init__()
        self.save_hyperparameters(args)
        self.model = GPT2LMHeadModel.from_pretrained(args.model_path)
        logging.info("Model loaded..")
        self.tokenizer = tokenizer
        self.setup_tokenizer()
        self.collator = collator

        self.rougescore = ROUGEScore(rouge_keys=('rougeL'), normalizer=lambda x: x)
        if self.hparams.label_smooth:
            self.loss_fct = LabelSmoothingCrossEntropy(smoothing=0.1)

    def setup(self, stage) -> None:
        if stage == 'fit':
            train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()

            # Calculate total steps
            tb_size = self.hparams.train_batchsize * max(1, self.trainer.gpus)
            ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
            self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def setup_tokenizer(self):
        self.vocab_size = len(self.tokenizer)
        self.model.resize_token_embeddings(len(self.tokenizer))

        assert self.tokenizer.pad_token_id == 50256
        assert self.tokenizer.sep_token_id == 50257  # add sep and eos
        assert self.tokenizer.eos_token_id == 50258  # add sep and eos

    def configure_optimizers(self):
        return configure_optimizers(self)

    def comput_metrix(self, logits, labels):
        """rewrite"""
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float()) / labels.size()[0]
        return acc

    def training_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = output.loss
        if self.hparams.label_smooth > 0:
            loss = self.loss_fct(output.logits.view(-1, self.vocab_size), batch["labels"].view(-1))

        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # get loss
        # batch_label = torch.where(batch["labels"] != -100, batch["labels"], self.tokenizer.pad_token_id)
        output = self.model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch["labels"])

        acc = self.comput_metrix(output.logits, batch["labels"])
        self.log('val_loss', output.loss, sync_dist=True)
        self.log('val_acc', acc, sync_dist=True)
        # self.log('val_ppl', torch.exp(output.loss), sync_dist=True)

        cond_output = self.generate(batch)

        # batch_label = torch.where(batch["labels"] != -100, batch["labels"], self.tokenizer.pad_token_id)
        pred = self.tokenizer.batch_decode(cond_output.sequences, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        ques = batch["target"]

        pred = [clean(white_space_fix(p)) for p in pred]
        for p, q in zip(pred, ques):
            print("pred in valid" + p)
            print("ques in valid" + q)

        pred = [chinese_char_tokenize(p) for p in pred]
        ques = [chinese_char_tokenize(q) for q in ques]
        self.rougescore.update(pred, ques)
        return pred

    def validation_epoch_end(self, validation_step_outputs):
        rouge = self.rougescore.compute()
        self.log('val_rouge', rouge["rougeL_fmeasure"], sync_dist=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.trainer._accelerator_connector.cluster_environment.global_rank() == 0:
            self.model.save_pretrained(os.path.join(
                self.trainer.checkpoint_callback.dirpath,
                'hf_pretrained_epoch{}_step{}'.format(checkpoint['epoch'], checkpoint['global_step'])))

    def generate(self, batch):
        # GreedySearchDecoderOnlyOutput(sequences=tensor([[seq1],[seq2]],device='cuda:0'), scores=None, attentions=None, hidden_states=None)
        return self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            return_dict_in_generate=True,
            output_scores=True,
            # max_length=self.args.text_length,
            do_sample=True,
            num_beams=5,
            # temperature = self.args.temp,
            # top_k=0,
            top_p=0.9,
            repetition_penalty=1.6,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=64,
        )

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            outputs = self.generate(batch)

        pred = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        pred = [clean(white_space_fix(p)) for p in pred]
        return pred


def save_prediction(pred_batches, test_data, args):
    os.makedirs(os.path.join(args.default_root_dir, "test"), exist_ok=True)
    newpred_file = os.path.join(os.path.join(args.default_root_dir, "test"), args.test_model+'_'+args.pred_file)
    f = open(newpred_file, 'a', encoding='utf-8')
    for batch_idx, data in enumerate(test_data):
        kno_batch, src_batch, tgt_batch = data["knowledge"], data["source"], data["target"]
        pred_batch = pred_batches[batch_idx]
        for idx in range(len(kno_batch)):
            output = {
                "knowledge": kno_batch[idx],
                "source": src_batch[idx],
                "target": tgt_batch[idx],
                "answer": pred_batch[idx]
            }
            json.dump(output, f, ensure_ascii=False)
            f.write('\n')


def get_tokenizer(args):
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, extra_ids=0)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, extra_ids=0)

    # has add into wenzhong 110M
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    print("We have added pad tokens")

    tokenizer.add_special_tokens({'sep_token': '<sep>'})
    print("We have added sep tokens")

    tokenizer.add_special_tokens({'eos_token': '<eos>'})
    print("We have added eos tokens")

    # If these tokens are already part of the vocabulary, it just let the Tokenizer know about them. If they don’t exist, the Tokenizer creates them, giving them a new id.
    # multitask
    if args.collator == "multitask":
        new_sp_token = {'additional_special_tokens': ['USER', 'TS']}  # USER TS in vocab
        tokenizer.add_special_tokens(new_sp_token)
        print("We have added user and ts tokens")

    return tokenizer


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--do_eval_only', action='store_true', default=False)
    args_parser.add_argument('--tokenizer_path', type=str, default=None)
    args_parser.add_argument('--collator', type=str, default='dial')
    args_parser.add_argument('--tensorboard_dir', type=str, default='gpt')
    args_parser.add_argument('--test_model', type=str, default='last.ckpt')
    args_parser.add_argument('--pred_file', type=str, default='last.ckpt')
    args_parser.add_argument('--qa_type', type=int, default=3)
    args_parser.add_argument('--sample_num', type=int, default=0)
    args_parser.add_argument('--deepspeed')
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = GPT2Finetuner.add_model_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args_parser = Collator.add_data_specific_args(args_parser)

    args = args_parser.parse_args()

    tokenizer = get_tokenizer(args)
    collator = Collator(tokenizer, args)
    data_module = UniversalDataModule(tokenizer=tokenizer, args=args, collate_fn=collator)

    module = GPT2Finetuner(tokenizer, collator, args)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = loggers.TensorBoardLogger(save_dir=args.tensorboard_dir, name="tf_log")
    checkpoint_callback = UniversalCheckpoint(args)
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback]
    )
    if not args.do_eval_only:
        logging.info("Begin training")
        trainer.fit(module, data_module)
    else:
        test_data = data_module.test_dataloader()
        preds_batches = trainer.predict(module, test_data, ckpt_path=os.path.join(args.save_ckpt_path, args.test_model))
        save_prediction(preds_batches, test_data, args)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error("Main program error:")
        logging.error(e)
        logging.error(traceback.format_exc())
