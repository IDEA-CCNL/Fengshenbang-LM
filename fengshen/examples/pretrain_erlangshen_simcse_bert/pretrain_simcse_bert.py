import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import torch.nn.functional as F
import argparse
import pytorch_lightning as pl
from dataclasses import dataclass
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)
from pytorch_lightning import (
    Trainer,
)
from fengshen.models.model_utils import (
    add_module_args,
    configure_optimizers,
    get_total_steps,
)
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    BertConfig)
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from torch.utils.data import default_collate


@dataclass
class SimCSEDataCollator:
    
    tokenizer: None  # 分词
    max_seq_length: 512
    z1: str = 'sent0'
    z2: str = 'sent1'
    z3: str = 'hard-negative'
    do_mlm: bool = False
    training_mode: str = 'sup'
    mlm_probability: float = 0.15

    def setup(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def get_mask(self,inputs_str):
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. 
            huggingface 开源的基本bert mask策略
        """
        input_ids = []
        labels = []
        for sent in inputs_str:
            inputs = self.tokenizer(sent,return_tensors='pt',add_special_tokens=False)['input_ids']
            label = inputs.clone()
            # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
            masked_indices = torch.bernoulli(torch.full(label.shape, self.mlm_probability)).bool()
            label[~masked_indices] = -100  # We only compute loss on masked tokens
            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(label.shape, 0.8)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(label.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), label.shape, dtype=torch.long)
            try:
                inputs[indices_random] = random_words[indices_random]
            except:
                print(torch.bernoulli(torch.full(label.shape, 0.5)).bool() & masked_indices & ~indices_replaced)
                print(indices_random)
                print(random_words)
            # The rest of the time (10% of the time) we keep the masked input tokens unchanged
            # add cls sep、token and pading
            # [b,seq_length] --> [b,max_seq_length]
            cls_token_id = torch.LongTensor([[self.tokenizer.cls_token_id]])
            sep_token_id = torch.LongTensor([[self.tokenizer.sep_token_id]])
            pad_token_id = torch.LongTensor([[self.tokenizer.pad_token_id]*(self.max_seq_length-inputs.shape[-1]-2)])
            special_token_unmask = torch.LongTensor([[-100]])
            # torch.cat([Tensor([[1,2,3]]),Tensor([[0]])],dim=1) --> Tensor([[0,1,2,3]])
            # 超过max_seq_length截断，反之padding 至少要添加cls和sep所以max_seq_length -2
            if inputs.shape[-1] >= (self.max_seq_length -2):
                inputs = inputs[:,:(self.max_seq_length-2)]
                label = label[:,:(self.max_seq_length-2)]
                input_ids.append(torch.cat([cls_token_id,inputs,sep_token_id],dim=1))
                labels.append(torch.cat([special_token_unmask,label,special_token_unmask],dim=1))
            else:
                input_ids.append(torch.cat([cls_token_id,inputs,sep_token_id,pad_token_id],dim=1))
                labels.append(torch.cat([special_token_unmask,label,special_token_unmask,torch.LongTensor([[-100]*(self.max_seq_length-inputs.shape[-1]-2)])],dim=1))
        return torch.stack(input_ids,dim=1).squeeze(), torch.stack(labels,dim=1).squeeze()

    def __call__(self,samples):
        """
        samples: sent0,sent1
        """
        samples = default_collate(samples) # {'sent0':[s1,s2]'sent1':[s3,s4],【hard-negative】}
        for k,v in samples.items():
            samples[k] = list(map(lambda x:x if x else '',v))
        # z1.input_ids: [b,max_seq_length]
        z1 = self.tokenizer.batch_encode_plus(samples[self.z1],max_length=self.max_seq_length,padding='max_length',truncation=True,return_tensors='pt')
        if self.training_mode == 'unsup':
            z2 = self.tokenizer.batch_encode_plus(samples[self.z1],max_length=self.max_seq_length,padding='max_length',truncation=True,return_tensors='pt')
        if self.training_mode == 'sup':
            z2 = self.tokenizer.batch_encode_plus(samples[self.z2],max_length=self.max_seq_length,padding='max_length',truncation=True,return_tensors='pt')
        if self.training_mode == 'hard-negative':
            z3 = self.tokenizer.batch_encode_plus(samples[self.z3],max_length=self.max_seq_length,padding='max_length',truncation=True,return_tensors='pt')
        inputs = dict()
        for k in z1.keys():
            if self.training_mode != 'hard-negative':
                inputs[k] = torch.stack([z1[k],z2[k]],dim=1)
            else:
                inputs[k] = torch.stack([z1[k],z2[k],z3[k]],dim=1)
        if self.do_mlm:
            # get mask text 
            z1_mask_inputs_ids,z1_labels = self.get_mask(samples[self.z1])
            z2_labels,z3_labels = None,None
            z2_mask_inputs_ids,z3_mask_inputs_ids = None,None
            if self.training_mode != 'unsup':
                z2_mask_inputs_ids,z2_labels = self.get_mask(samples[self.z2])
            if self.training_mode == 'hard-negative':
                z3_mask_inputs_ids,z3_labels = self.get_mask(samples[self.z3])
            if (z2_labels is not None) and (z3_labels is not None):
                mask_inputs_ids = torch.stack([z1_mask_inputs_ids,z2_mask_inputs_ids,z3_mask_inputs_ids],dim=1)
                labels = torch.stack([z1_labels,z2_labels,z3_labels],dim=1)
            elif z2_labels is not None:
                mask_inputs_ids = torch.stack([z1_mask_inputs_ids,z2_mask_inputs_ids],dim=1)
                labels = torch.stack([z1_labels,z2_labels],dim=1)
            else:
                labels = z1_labels
                mask_inputs_ids = z1_mask_inputs_ids
            inputs['labels'] = labels
            inputs['mask_inputs_ids'] = mask_inputs_ids
        return inputs


class SimCSEModel(pl.LightningModule):

    @staticmethod
    def add_module_specific_args(parent_args):
        parser = parent_args.add_argument_group('BaseModel')
        parser.add_argument('--max_seq_length', type=int, default=512)
        parser.add_argument('--sample_content_key', type=str,default=['sent0','sent1'],nargs='+')
        parser.add_argument('--training-mode', type=str, default='sup', choices=['unsup','sup','hard-negative'])
        parser.add_argument('--pooling', type=str,default='cls', choices=['cls','pooler','last-avg','first-last-avg'])
        parser.add_argument('--temp', type=float,default=0.05)
        parser.add_argument('--reinitialize_model', type=bool, default=False)
        parser.add_argument('--do-mlm', action='store_true',help="if do mask task,should open this args")
        return parent_args

    def __init__(self, args, tokenizer):
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        print('^_^ args.reinitialize_model',args.reinitialize_model)
        if args.reinitialize_model:
            print('prtrain from a empty model')
            self.config = BertConfig.from_pretrained(args.model_path)
            self.model = BertForMaskedLM(self.config)
        else:
            print('prtrain from a trained model')
            self.model = BertForMaskedLM.from_pretrained(args.model_path)

    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            print(f'batch size: {self.hparams.train_batchsize}')
            print(f'world size: {self.trainer.world_size}')
            print(f'accumulate_grad_batches: {self.trainer.accumulate_grad_batches}')
            print('Total steps: {}' .format(self.total_steps))

    def training_step(self, batch, batch_idx):
        # batch.input_ids: [b,num_sent,seq_length] 
        b,num_sent,seq_length = batch['input_ids'].shape
        input_ids = batch['input_ids'].view(-1,seq_length) # [b*num,seq_length]
        attention_mask = batch['attention_mask'].view(-1,seq_length)
        token_type_ids = batch['token_type_ids'].view(-1,seq_length)

        outputs = self.model(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,output_hidden_states=True)
        # 计算句子表征 [b*num_sent,hidden_state]
        if self.hparams.pooling == 'first-last-avg':
            sentences_embedding = torch.mean(outputs.hidden_states[0]+outputs.hidden_states[-1],dim=1).squeeze()
        elif self.hparams.pooling == 'last-avg':
            sentences_embedding = torch.mean(outputs.hidden_states[-1],dim=1).squeeze()
        elif self.hparams.pooling == 'cls':
            # 取最后一层的每个句子的第一个token，cls
            sentences_embedding = outputs.hidden_states[-1][:,0,:].squeeze()
        elif self.hparams.pooling == 'pooler':
            output = torch.mean(outputs.hidden_states[-1],dim=1)
            sentences_embedding =  F.gelu(F.linear(output))
        # sentences_embedding:[B,num_sent,hidden_size]
        sentences_embedding = sentences_embedding.view(b,num_sent,sentences_embedding.size(-1))
        # 计算similarity loss
        # sim：[b,b*(num_sent-1)],没有计算z1互相间的相似度
        z1 = sentences_embedding[:,0,:]
        z2 = sentences_embedding[:,1,:]
        sim = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)/self.hparams.temp
        if sentences_embedding.shape[1] == 3:
            # hard negtive
            z3 = sentences_embedding[:,2,:]
            no_sim = F.cosine_similarity(z1.unsqueeze(1), z3.unsqueeze(0), dim=-1)/self.hparams.temp
            sim = torch.cat([sim, no_sim],dim=1) 
        similarity_labels = torch.arange(sentences_embedding.shape[0]).to(self.device)
        loss = F.cross_entropy(sim,similarity_labels)
        # do mlm
        if self.hparams.do_mlm:
            labels = batch['labels'].view(-1,seq_length)
            mask_inputs_ids = batch['mask_inputs_ids'].view(-1,seq_length)
            outputs = self.model(mask_inputs_ids,labels=labels,output_hidden_states=True)
            lm_loss = outputs.loss
            acc = self.comput_metrix(outputs.logits, labels)
            self.log('train_acc', acc, sync_dist=True)
            self.log('lm_loss', lm_loss,sync_dist=True)
            self.log('simcse_loss', loss,sync_dist=True)
            loss += lm_loss
        self.log('train_loss', loss,sync_dist=True)
        return loss

    def comput_metrix(self, logits, labels):
        ones = torch.ones_like(labels)
        zero = torch.zeros_like(labels)
        mask = torch.where(labels < 0, zero, ones)
        mask = mask.view(size=(-1,)).float()

        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        corr = torch.multiply(corr.float(), mask)
        acc = torch.sum(corr.float()) / torch.sum(mask)
        return acc

    def validation_step(self, batch, batch_idx):
        # print('now doing validation',batch_idx)
        # output = self.model(**batch)
        # acc = self.comput_metrix(output.logits, batch['labels'])
        # self.log('val_loss', output.loss,sync_dist=True)
        # self.log('val_acc', acc,sync_dist=True)
        pass

    def configure_optimizers(self):
        return configure_optimizers(self)


def main():
    args_parser = argparse.ArgumentParser("gpt pretrain")
    args_parser = add_module_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = SimCSEModel.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()
    print('args parse done')
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    collate_fn = SimCSEDataCollator(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        do_mlm=args.do_mlm,
        training_mode=args.training_mode,
    )
    collate_fn.setup()
    data_module = UniversalDataModule(tokenizer=tokenizer, args=args, collate_fn=collate_fn)
    # 将数据 分句拼接 变成1024长度的样本，如果数据不需要这样处理，可以忽略
    # print('---------mapping data---------')
    # data_module.datasets = data_module.datasets.map(map_fun,batched=True,
    #         num_proc=args.dataloader_workers,
    #         remove_columns=data_module.datasets.column_names['train'],
    #         load_from_cache_file=True)
    # print(data_module.datasets)
    # print('   ------check data------   ')
    # for i in range(10):
    #     print(f"train data {i}:\n{data_module.datasets['train'][i]}")
    print('data load done')
    model = SimCSEModel(args, tokenizer=tokenizer)
    print('model init done')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)

    if args.load_ckpt_path is not None and not os.path.exists(args.load_ckpt_path):
        print('--------warning no checkpoint found--------, remove args')
        args.load_ckpt_path = None
    #limit_train_batches=1,limit_val_batches=1,limit_test_batches=1, 值为整数时，每个epoch🈯只过指定个数的batchse数
    trainer = Trainer.from_argparse_args(args,limit_train_batches=0.1,limit_val_batches=1.0,limit_test_batches=1.0,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    trainer.fit(model, data_module, ckpt_path=args.load_ckpt_path)


if __name__ == '__main__':
    main()
