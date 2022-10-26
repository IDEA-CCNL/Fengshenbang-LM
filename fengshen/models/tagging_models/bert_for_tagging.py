import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.crf import CRF
from .layers.bert_output import BiaffineClassifierOutput, TokenClassifierOutput, SpanClassifierOutput
from transformers import BertPreTrainedModel
from transformers import BertModel
from .layers.linears import PoolerEndLogits, PoolerStartLogits, Biaffine
from torch.nn import CrossEntropyLoss
from .losses.focal_loss import FocalLoss
from .losses.label_smoothing import LabelSmoothingCrossEntropy

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'IDEA-CCNL/BertCrf': '/cognitive_comp/lujunyu/NER/outputs/ccks_crf/bert/best_checkpoint/pytorch_model.bin',
}
PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'IDEA-CCNL/BertCrf': '/cognitive_comp/lujunyu/NER/outputs/ccks_crf/bert/best_checkpoint/config.json',
}

class BertLinear(BertPreTrainedModel):
    def __init__(self, config, num_labels, loss_type):
        super(BertLinear, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.loss_type = loss_type


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, input_len=None,
                position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,output_hidden_states=True)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss=None

        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(loss, logits)  # (loss), scores, (hidden_states), (attentions)
        
class BertCrf(BertPreTrainedModel):
    def __init__(self, config, num_labels, loss_type):
        super(BertCrf, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None,input_len=None):
        outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss=None
        if labels is not None:
            loss = -1 * self.crf(emissions = logits, tags=labels, mask=attention_mask)

        return TokenClassifierOutput(loss, logits)

class BertBiaffine(BertPreTrainedModel):
    def __init__(self, config, num_labels, loss_type):
        super(BertBiaffine, self).__init__(config)
        self.num_labels=num_labels
        self.bert = BertModel(config)
        self.start_layer=torch.nn.Sequential(torch.nn.Linear(in_features=config.hidden_size, out_features=128),torch.nn.ReLU())
        self.end_layer=torch.nn.Sequential(torch.nn.Linear(in_features=config.hidden_size, out_features=128),torch.nn.ReLU())
        self.biaffne_layer = Biaffine(128,self.num_labels)

        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size//2, num_layers=2, dropout=0.5,
                             batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.loss_type = loss_type
                
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, span_labels=None, span_mask=None, input_len=None):
        outputs=self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output=outputs[0]
        sequence_output=self.dropout(self.lstm(sequence_output)[0])

        start_logits=self.start_layer(sequence_output)
        end_logits=self.end_layer(sequence_output)

        span_logits=self.biaffne_layer(start_logits,end_logits)
        # breakpoint()
        span_loss=None
        if span_labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            
            span_logits=span_logits.contiguous()

            active_loss=span_mask.view(-1) == 1
            active_logits = span_logits.view(-1, self.num_labels)[active_loss]
            active_labels = span_labels.view(-1)[active_loss]
            span_loss = 10*loss_fct(active_logits, active_labels)

        return BiaffineClassifierOutput(loss=span_loss,span_logits=span_logits)


class BertSpan(BertPreTrainedModel):
    def __init__(self, config, num_labels, loss_type, soft_label=True):
        super(BertSpan, self).__init__(config)
        self.soft_label = soft_label
        self.num_labels = num_labels
        self.loss_type = loss_type
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None,end_positions=None, subjects=None, input_len=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)

        total_loss=None
        if start_positions is not None and end_positions is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type =='lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()

            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits.view(-1, self.num_labels)[active_loss]
            active_end_logits = end_logits.view(-1, self.num_labels)[active_loss]

            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_labels = end_positions.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2

        return SpanClassifierOutput(loss=total_loss,start_logits=start_logits,end_logits=end_logits)


# class BertLstmCrf(BertPreTrainedModel):
#     def __init__(self, config):
#         super(BertLstmCrf, self).__init__(config)
#         self.bert = BertModel(config)
#         self.lstm = nn.LSTM(config.hidden_size, config.hidden_size//2, 2,
#                              batch_first=True, bidirectional=True)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#         self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
#         self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
#         self.crf = CRF(num_tags=config.num_labels, batch_first=True)
#         self.layernorm = nn.LayerNorm(config.hidden_size)

#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, input_lens=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         sequence_output =outputs[0]

#         sequence_output = self.dropout1(sequence_output)
#         sequence_output = self.dropout2(self.lstm(sequence_output)[0])
#         logits = self.classifier(self.layernorm(sequence_output))
#         outputs = (logits,)
#         if labels is not None:
#             loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
#             outputs = (-1 * loss,) + outputs
#         return outputs