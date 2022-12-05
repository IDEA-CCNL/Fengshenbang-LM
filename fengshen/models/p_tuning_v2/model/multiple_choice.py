import torch
from torch._C import NoopLogger
import torch.nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers import BertModel, BertPreTrainedModel, RobertaTokenizer
from transformers import RobertaModel, RobertaPreTrainedModel

from transformers.modeling_outputs import MultipleChoiceModelOutput, BaseModelOutput, Seq2SeqLMOutput

from model.prefix_encoder import PrefixEncoder, ExplPrefixEncoder
from model.deberta import DebertaModel, DebertaPreTrainedModel, ContextPooler, StableDropout
from model.debertaV2 import DebertaV2Model, DebertaV2PreTrainedModel

from transformers import AlbertPreTrainedModel
from transformers import AlbertModel
import pdb


class BertForMultipleChoice(BertPreTrainedModel):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
    input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
    token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_choices = 2

    model = BertForMultipleChoice(config, num_choices)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, num_choices = input_ids.shape[:2]

        input_ids = input_ids.reshape(-1, input_ids.size(-1))
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.size(-1))
        attention_mask = attention_mask.reshape(-1, attention_mask.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertPrefixForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)

        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        bert_param = 0
        for name, param in self.bert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param)) # 9860105
    
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, num_choices = input_ids.shape[:2] if input_ids is not None else inputs_embeds[:2]

        input_ids = input_ids.reshape(-1, input_ids.size(-1)) if input_ids is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.size(-1)) if attention_mask is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        past_key_values = self.get_prompt(batch_size=batch_size * num_choices)
        prefix_attention_mask = torch.ones(batch_size * num_choices, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaPrefixForMultipleChoice(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)

        self.init_weights()


        for param in self.roberta.parameters():
            param.requires_grad = False
            
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, num_choices = input_ids.shape[:2] if input_ids is not None else inputs_embeds.shape[:2]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        past_key_values = self.get_prompt(batch_size=batch_size * num_choices)
        prefix_attention_mask = torch.ones(batch_size * num_choices, self.pre_seq_len).to(self.roberta.device)
        flat_attention_mask = torch.cat((prefix_attention_mask, flat_attention_mask), dim=1)

        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class RobertaMultiExplPrefixForMultipleChoice(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.prefix_fusion_way = config.prefix_fusion_way
        self.roberta = RobertaModel(config)
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)
        # self.merged_classifier = torch.nn.Linear(2*config.hidden_size, 1)
        self.init_weights()


        for param in self.roberta.parameters():
            param.requires_grad = False
            
        self.pre_seq_len = config.pre_seq_len
        self.nle_prefix_len = config.nle_prefix_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        self.used_triplets_type = config.used_triplets_type

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.task_prefix_encoder = PrefixEncoder(config)
        
        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))
        print('all param is {}'.format(all_param))

    def get_prompt(self, batch_size, nle_hidden_states):
        nle_hidden_states = nle_hidden_states.view(batch_size, 1, -1).repeat(1, self.nle_prefix_len,1)
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        expl_past_key_values = self.expl_prefix_encoder(nle_hidden_states)
        #past_key_values = torch.cat([past_key_values, expl_past_key_values], dim=1)
        fusion_prefix_len = self.pre_seq_len
        if self.prefix_fusion_way == 'concat':
            #pdb.set_trace()
            past_key_values = torch.cat([past_key_values, expl_past_key_values], dim=1)
            fusion_prefix_len = self.pre_seq_len + self.nle_prefix_len
        elif self.prefix_fusion_way == 'add':
            #pdb.set_trace()
            past_key_values = (past_key_values + expl_past_key_values)/2
            fusion_prefix_len = self.pre_seq_len
        elif self.prefix_fusion_way == 'none' or config.prefix:
            past_key_values = past_key_values
            fusion_prefix_len = self.pre_seq_len
        past_key_values = past_key_values.view(
            batch_size,
            fusion_prefix_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_task_prompt(self, batch_size, knowledge_embeddings):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        # pdb.set_trace()
        if self.prefix_fusion_way == 'concat':
            # pdb.set_trace()
            past_key_values = self.task_prefix_encoder(prefix_tokens, knowledge_embeddings)
            past_key_values = past_key_values.view(
                batch_size,
                self.pre_seq_len,
                self.n_layer * 2, 
                self.n_head,
                self.n_embd
            )
        else:
            # pdb.set_trace()
            past_key_values = self.task_prefix_encoder(prefix_tokens)
            past_key_values = past_key_values.view(
                batch_size,
                self.pre_seq_len,
                self.n_layer * 2, 
                self.n_head,
                self.n_embd
            )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
        self,

        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,

        labels=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

        combine_selected_individual_view_triplets_input_ids=None,
        combine_selected_individual_view_triplets_token_type_ids=None,
        combine_selected_individual_view_triplets_attention_mask=None,
        combine_selected_individual_view_triplets_position_ids=None,

        combine_individual_view_triplets_input_ids=None,
        combine_individual_view_triplets_token_type_ids=None,
        combine_individual_view_triplets_attention_mask=None,
        combine_individual_view_triplets_position_ids=None,

        combine_latent_view_triplets_input_ids=None,
        combine_latent_view_triplets_token_type_ids=None,
        combine_latent_view_triplets_attention_mask=None,
        combine_latent_view_triplets_position_ids=None,

        combine_group_view_triplets_input_ids=None,
        combine_group_view_triplets_token_type_ids=None,
        combine_group_view_triplets_attention_mask=None,
        combine_group_view_triplets_position_ids=None,

        combine_retri_view_triplets_input_ids=None,
        combine_retri_view_triplets_token_type_ids=None,
        combine_retri_view_triplets_attention_mask=None,
        combine_retri_view_triplets_position_ids=None,

        combine_meaning_view_triplets_input_ids=None,
        combine_meaning_view_triplets_token_type_ids=None,
        combine_meaning_view_triplets_attention_mask=None,
        combine_meaning_view_triplets_position_ids=None,

        selected_individual_view_triplets_input_ids=None,
        selected_individual_view_triplets_token_type_ids=None,
        selected_individual_view_triplets_attention_mask=None,
        selected_individual_view_triplets_position_ids=None,

        individual_view_triplets_input_ids=None,
        individual_view_triplets_token_type_ids=None,
        individual_view_triplets_attention_mask=None,
        individual_view_triplets_position_ids=None,

        latent_view_triplets_input_ids=None,
        latent_view_triplets_token_type_ids=None,
        latent_view_triplets_attention_mask=None,
        latent_view_triplets_position_ids=None,

        group_view_triplets_input_ids=None,
        group_view_triplets_token_type_ids=None,
        group_view_triplets_attention_mask=None,
        group_view_triplets_position_ids=None,
        
        tv_individual_group_triplets_input_ids=None,
        tv_individual_group_triplets_token_type_ids=None,
        tv_individual_group_triplets_attention_mask=None,
        tv_individual_group_triplets_position_ids=None,

        tv_individual_retri_triplets_input_ids=None,
        tv_individual_retri_triplets_token_type_ids=None,
        tv_individual_retri_triplets_attention_mask=None,
        tv_individual_retri_triplets_position_ids=None,

        tv_individual_meaning_triplets_input_ids=None,
        tv_individual_meaning_triplets_token_type_ids=None,
        tv_individual_meaning_triplets_attention_mask=None,
        tv_individual_meaning_triplets_position_ids=None,

        tv_individual_latent_triplets_input_ids=None,
        tv_individual_latent_triplets_token_type_ids=None,
        tv_individual_latent_triplets_attention_mask=None,
        tv_individual_latent_triplets_position_ids=None,

        tv_group_latent_triplets_input_ids=None,
        tv_group_latent_triplets_token_type_ids=None,
        tv_group_latent_triplets_attention_mask=None,
        tv_group_latent_triplets_position_ids=None,

        v3_individual_group_triplets_input_ids=None,
        v3_individual_group_triplets_token_type_ids=None,
        v3_individual_group_triplets_attention_mask=None,
        v3_individual_group_triplets_position_ids=None,

        v3_individual_retri_meaning_triplets_input_ids=None,
        v3_individual_retri_meaning_triplets_token_type_ids=None,
        v3_individual_retri_meaning_triplets_attention_mask=None,
        v3_individual_retri_meaning_triplets_position_ids=None,

    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, num_choices = input_ids.shape[:2] if input_ids is not None else inputs_embeds.shape[:2]

        # individual_view_triplets_input_ids=None,
        # individual_view_triplets_token_type_ids=None,
        # individual_view_triplets_attention_mask=None,
        # individual_view_triplets_position_ids=None,

        # group_view_triplets_input_ids=None,
        # group_view_triplets_token_type_ids=None,
        # group_view_triplets_attention_mask=None,
        # group_view_triplets_position_ids=None,

        # flat_individual_view_triplets_input_ids = individual_view_triplets_input_ids.view(-1, individual_view_triplets_input_ids.size(-1)) if individual_view_triplets_input_ids is not None else None
        # flat_individual_view_triplets_position_ids = individual_view_triplets_position_ids.view(-1, individual_view_triplets_position_ids.size(-1)) if individual_view_triplets_position_ids is not None else None
        # flat_individual_view_triplets_token_type_ids = individual_view_triplets_token_type_ids.view(-1, individual_view_triplets_token_type_ids.size(-1)) if individual_view_triplets_token_type_ids is not None else None
        # individual_view_triplets_embeddings = self.roberta.embeddings(input_ids=flat_individual_view_triplets_input_ids, 
        #                                                               token_type_ids=flat_individual_view_triplets_token_type_ids,
        #                                                               position_ids=flat_individual_view_triplets_position_ids)
        # flat_group_view_triplets_input_ids = group_view_triplets_input_ids.view(-1, group_view_triplets_input_ids.size(-1)) if group_view_triplets_input_ids is not None else None
        # flat_group_view_triplets_position_ids = group_view_triplets_position_ids.view(-1, group_view_triplets_position_ids.size(-1)) if group_view_triplets_position_ids is not None else None
        # flat_group_view_triplets_token_type_ids = group_view_triplets_token_type_ids.view(-1, group_view_triplets_token_type_ids.size(-1)) if group_view_triplets_token_type_ids is not None else None
        # group_view_triplets_embeddings = self.roberta.embeddings(input_ids=flat_group_view_triplets_input_ids, 
        #                                                               token_type_ids=flat_group_view_triplets_token_type_ids,
        #                                                               position_ids=flat_group_view_triplets_position_ids)
        # pdb.set_trace()
        # past_key_values = self.get_task_prompt(batch_size=batch_size * num_choices, knowledge_embeddings = torch.stack([individual_view_triplets_embeddings.mean(axis=1), group_view_triplets_embeddings.mean(axis=1)], axis=1))
        relation_text = [
    'is the antonym of',
    'is at location of',
    'is capable of',
    'causes',
    'is created by',
    'is a kind of',
    'desires',
    'has subevent',
    'is part of',
    'has context',
    'has property',
    'is made of',
    'is not capable of',
    'does not desires',
    'is',
    'is related to',
    'is used for',
]
        rel_mapping = {
                        'CausesDesire': "causes desire",
                        'HasProperty': "has property",
                        'CapableOf': 'capable of',
                        'PartOf': 'part of',
                        'AtLocation': 'at location',
                        'Desires': 'desires',
                        'HasPrerequisite': 'has prerequisite',
                        'HasSubevent': 'has subevent',
                        'Antonym': 'antonym',
                        'Causes': 'causes',
        }
        #relation_text = rel_mapping.values()

        knowledge_embeddings = []
        for rel in relation_text:
            # pdb.set_trace()
            rel_dict = self.roberta_tokenizer(rel, padding=True, max_length=7, truncation=True)
            rel_emb = self.roberta.embeddings(input_ids=torch.tensor(rel_dict['input_ids']).unsqueeze(0).to(self.roberta.device))
            knowledge_embeddings.append(rel_emb.mean(axis=1))
        # pdb.set_trace()
        knowledge_embeddings = None#torch.stack(knowledge_embeddings, axis=1)
        # pdb.set_trace()
        past_key_values = self.get_task_prompt(batch_size=batch_size * num_choices, knowledge_embeddings = knowledge_embeddings)
        #pdb.set_trace()
        if self.used_triplets_type == "individual":
            input_ids=combine_individual_view_triplets_input_ids
            token_type_ids=combine_individual_view_triplets_token_type_ids
            attention_mask=combine_individual_view_triplets_attention_mask
            position_ids=combine_individual_view_triplets_position_ids
        elif self.used_triplets_type == "selected_individual":
            input_ids=combine_selected_individual_view_triplets_input_ids
            token_type_ids=combine_selected_individual_view_triplets_token_type_ids
            attention_mask=combine_selected_individual_view_triplets_attention_mask
            position_ids=combine_selected_individual_view_triplets_position_ids
        elif self.used_triplets_type == "latent":
            input_ids=combine_latent_view_triplets_input_ids
            token_type_ids=combine_latent_view_triplets_token_type_ids
            attention_mask=combine_latent_view_triplets_attention_mask
            position_ids=combine_latent_view_triplets_position_ids
        elif self.used_triplets_type == "group":
            input_ids=combine_group_view_triplets_input_ids
            token_type_ids=combine_group_view_triplets_token_type_ids
            attention_mask=combine_group_view_triplets_attention_mask
            position_ids=combine_group_view_triplets_position_ids
        elif self.used_triplets_type == "retri":
            # print("****used_triplets_type == retri****")
            input_ids=combine_retri_view_triplets_input_ids
            token_type_ids=combine_retri_view_triplets_token_type_ids
            attention_mask=combine_retri_view_triplets_attention_mask
            position_ids=combine_retri_view_triplets_position_ids
        elif self.used_triplets_type == "meaning":
            # print("****used_triplets_type == retri****")
            input_ids=combine_meaning_view_triplets_input_ids
            token_type_ids=combine_meaning_view_triplets_token_type_ids
            attention_mask=combine_meaning_view_triplets_attention_mask
            position_ids=combine_meaning_view_triplets_position_ids
        elif self.used_triplets_type == "individual_group":
            # pdb.set_trace()
            input_ids=tv_individual_group_triplets_input_ids
            token_type_ids=tv_individual_group_triplets_token_type_ids
            attention_mask=tv_individual_group_triplets_attention_mask
            position_ids=tv_individual_group_triplets_position_ids
        elif self.used_triplets_type == "individual_retri":
            input_ids=tv_individual_retri_triplets_input_ids
            token_type_ids=tv_individual_retri_triplets_token_type_ids
            attention_mask=tv_individual_retri_triplets_attention_mask
            position_ids=tv_individual_retri_triplets_position_ids
        elif self.used_triplets_type == "individual_meaning":
            input_ids=tv_individual_meaning_triplets_input_ids
            token_type_ids=tv_individual_meaning_triplets_token_type_ids
            attention_mask=tv_individual_meaning_triplets_attention_mask
            position_ids=tv_individual_meaning_triplets_position_ids
        elif self.used_triplets_type == "individual_latent":
            input_ids=tv_individual_group_triplets_input_ids
            token_type_ids=tv_individual_latent_triplets_token_type_ids
            attention_mask=tv_individual_latent_triplets_attention_mask
            position_ids=tv_individual_latent_triplets_position_ids
        elif self.used_triplets_type == "group_individual":
            input_ids=tv_group_individual_triplets_input_ids
            token_type_ids=tv_group_individual_triplets_token_type_ids
            attention_mask=tv_group_individual_triplets_attention_mask
            position_ids=tv_group_individual_triplets_position_ids
        elif self.used_triplets_type == "all":
            input_ids=v3_individual_group_triplets_input_ids
            token_type_ids=v3_individual_group_triplets_token_type_ids
            attention_mask=v3_individual_group_triplets_attention_mask
            position_ids=v3_individual_group_triplets_position_ids
        elif self.used_triplets_type == "individual_retri_meaning":
            input_ids=v3_individual_retri_meaning_triplets_input_ids
            token_type_ids=v3_individual_retri_meaning_triplets_token_type_ids
            attention_mask=v3_individual_retri_meaning_triplets_attention_mask
            position_ids=v3_individual_retri_meaning_triplets_position_ids


        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        prefix_attention_mask = torch.ones(batch_size * num_choices, past_key_values[0].size(3)).to(self.roberta.device)
        flat_attention_mask = torch.cat((prefix_attention_mask, flat_attention_mask), dim=1)
        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        # outputs = self.roberta(
        #     flat_input_ids,
        #     position_ids=flat_position_ids,
        #     token_type_ids=flat_token_type_ids,
        #     attention_mask=flat_attention_mask,
        #     head_mask=head_mask,
        #     inputs_embeds=flat_inputs_embeds,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict
        # )
        pooled_output = self.dropout(outputs[1])
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AlbertMultiExplPrefixForMultipleChoice(AlbertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.prefix_fusion_way = config.prefix_fusion_way
        #self.roberta = RobertaModel(config)
        self.albert = AlbertModel(config)
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)
        # self.merged_classifier = torch.nn.Linear(2*config.hidden_size, 1)
        self.init_weights()


        for param in self.albert.parameters():
            param.requires_grad = False
            
        self.pre_seq_len = config.pre_seq_len
        self.nle_prefix_len = config.nle_prefix_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        self.used_triplets_type = config.used_triplets_type

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.task_prefix_encoder = PrefixEncoder(config)
        
        bert_param = 0
        for name, param in self.albert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))
        print('all param is {}'.format(all_param))

    def get_prompt(self, batch_size, nle_hidden_states):
        nle_hidden_states = nle_hidden_states.view(batch_size, 1, -1).repeat(1, self.nle_prefix_len,1)
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        expl_past_key_values = self.expl_prefix_encoder(nle_hidden_states)
        #past_key_values = torch.cat([past_key_values, expl_past_key_values], dim=1)
        fusion_prefix_len = self.pre_seq_len
        if self.prefix_fusion_way == 'concat':
            #pdb.set_trace()
            past_key_values = torch.cat([past_key_values, expl_past_key_values], dim=1)
            fusion_prefix_len = self.pre_seq_len + self.nle_prefix_len
        elif self.prefix_fusion_way == 'add':
            #pdb.set_trace()
            past_key_values = (past_key_values + expl_past_key_values)/2
            fusion_prefix_len = self.pre_seq_len
        elif self.prefix_fusion_way == 'none' or config.prefix:
            past_key_values = past_key_values
            fusion_prefix_len = self.pre_seq_len
        past_key_values = past_key_values.view(
            batch_size,
            fusion_prefix_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_task_prompt(self, batch_size, knowledge_embeddings):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.albert.device)
        # pdb.set_trace()
        if self.prefix_fusion_way == 'concat':
            # pdb.set_trace()
            past_key_values = self.task_prefix_encoder(prefix_tokens, knowledge_embeddings)
            past_key_values = past_key_values.view(
                batch_size,
                self.pre_seq_len,
                self.n_layer * 2, 
                self.n_head,
                self.n_embd
            )
        else:
            # pdb.set_trace()
            past_key_values = self.task_prefix_encoder(prefix_tokens)
            past_key_values = past_key_values.view(
                batch_size,
                self.pre_seq_len,
                self.n_layer * 2, 
                self.n_head,
                self.n_embd
            )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
        self,

        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,

        labels=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,

        combine_selected_individual_view_triplets_input_ids=None,
        combine_selected_individual_view_triplets_token_type_ids=None,
        combine_selected_individual_view_triplets_attention_mask=None,
        combine_selected_individual_view_triplets_position_ids=None,

        combine_individual_view_triplets_input_ids=None,
        combine_individual_view_triplets_token_type_ids=None,
        combine_individual_view_triplets_attention_mask=None,
        combine_individual_view_triplets_position_ids=None,

        combine_latent_view_triplets_input_ids=None,
        combine_latent_view_triplets_token_type_ids=None,
        combine_latent_view_triplets_attention_mask=None,
        combine_latent_view_triplets_position_ids=None,

        combine_group_view_triplets_input_ids=None,
        combine_group_view_triplets_token_type_ids=None,
        combine_group_view_triplets_attention_mask=None,
        combine_group_view_triplets_position_ids=None,

        combine_retri_view_triplets_input_ids=None,
        combine_retri_view_triplets_token_type_ids=None,
        combine_retri_view_triplets_attention_mask=None,
        combine_retri_view_triplets_position_ids=None,

        combine_meaning_view_triplets_input_ids=None,
        combine_meaning_view_triplets_token_type_ids=None,
        combine_meaning_view_triplets_attention_mask=None,
        combine_meaning_view_triplets_position_ids=None,

        selected_individual_view_triplets_input_ids=None,
        selected_individual_view_triplets_token_type_ids=None,
        selected_individual_view_triplets_attention_mask=None,
        selected_individual_view_triplets_position_ids=None,

        individual_view_triplets_input_ids=None,
        individual_view_triplets_token_type_ids=None,
        individual_view_triplets_attention_mask=None,
        individual_view_triplets_position_ids=None,

        latent_view_triplets_input_ids=None,
        latent_view_triplets_token_type_ids=None,
        latent_view_triplets_attention_mask=None,
        latent_view_triplets_position_ids=None,

        group_view_triplets_input_ids=None,
        group_view_triplets_token_type_ids=None,
        group_view_triplets_attention_mask=None,
        group_view_triplets_position_ids=None,
        
        tv_individual_group_triplets_input_ids=None,
        tv_individual_group_triplets_token_type_ids=None,
        tv_individual_group_triplets_attention_mask=None,
        tv_individual_group_triplets_position_ids=None,

        tv_individual_retri_triplets_input_ids=None,
        tv_individual_retri_triplets_token_type_ids=None,
        tv_individual_retri_triplets_attention_mask=None,
        tv_individual_retri_triplets_position_ids=None,

        tv_individual_meaning_triplets_input_ids=None,
        tv_individual_meaning_triplets_token_type_ids=None,
        tv_individual_meaning_triplets_attention_mask=None,
        tv_individual_meaning_triplets_position_ids=None,

        tv_individual_latent_triplets_input_ids=None,
        tv_individual_latent_triplets_token_type_ids=None,
        tv_individual_latent_triplets_attention_mask=None,
        tv_individual_latent_triplets_position_ids=None,

        tv_group_latent_triplets_input_ids=None,
        tv_group_latent_triplets_token_type_ids=None,
        tv_group_latent_triplets_attention_mask=None,
        tv_group_latent_triplets_position_ids=None,

        v3_individual_group_triplets_input_ids=None,
        v3_individual_group_triplets_token_type_ids=None,
        v3_individual_group_triplets_attention_mask=None,
        v3_individual_group_triplets_position_ids=None,

        v3_individual_retri_meaning_triplets_input_ids=None,
        v3_individual_retri_meaning_triplets_token_type_ids=None,
        v3_individual_retri_meaning_triplets_attention_mask=None,
        v3_individual_retri_meaning_triplets_position_ids=None,

    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, num_choices = input_ids.shape[:2] if input_ids is not None else inputs_embeds.shape[:2]

        # individual_view_triplets_input_ids=None,
        # individual_view_triplets_token_type_ids=None,
        # individual_view_triplets_attention_mask=None,
        # individual_view_triplets_position_ids=None,

        # group_view_triplets_input_ids=None,
        # group_view_triplets_token_type_ids=None,
        # group_view_triplets_attention_mask=None,
        # group_view_triplets_position_ids=None,

        # flat_individual_view_triplets_input_ids = individual_view_triplets_input_ids.view(-1, individual_view_triplets_input_ids.size(-1)) if individual_view_triplets_input_ids is not None else None
        # flat_individual_view_triplets_position_ids = individual_view_triplets_position_ids.view(-1, individual_view_triplets_position_ids.size(-1)) if individual_view_triplets_position_ids is not None else None
        # flat_individual_view_triplets_token_type_ids = individual_view_triplets_token_type_ids.view(-1, individual_view_triplets_token_type_ids.size(-1)) if individual_view_triplets_token_type_ids is not None else None
        # individual_view_triplets_embeddings = self.roberta.embeddings(input_ids=flat_individual_view_triplets_input_ids, 
        #                                                               token_type_ids=flat_individual_view_triplets_token_type_ids,
        #                                                               position_ids=flat_individual_view_triplets_position_ids)
        # flat_group_view_triplets_input_ids = group_view_triplets_input_ids.view(-1, group_view_triplets_input_ids.size(-1)) if group_view_triplets_input_ids is not None else None
        # flat_group_view_triplets_position_ids = group_view_triplets_position_ids.view(-1, group_view_triplets_position_ids.size(-1)) if group_view_triplets_position_ids is not None else None
        # flat_group_view_triplets_token_type_ids = group_view_triplets_token_type_ids.view(-1, group_view_triplets_token_type_ids.size(-1)) if group_view_triplets_token_type_ids is not None else None
        # group_view_triplets_embeddings = self.roberta.embeddings(input_ids=flat_group_view_triplets_input_ids, 
        #                                                               token_type_ids=flat_group_view_triplets_token_type_ids,
        #                                                               position_ids=flat_group_view_triplets_position_ids)
        # pdb.set_trace()
        # past_key_values = self.get_task_prompt(batch_size=batch_size * num_choices, knowledge_embeddings = torch.stack([individual_view_triplets_embeddings.mean(axis=1), group_view_triplets_embeddings.mean(axis=1)], axis=1))
        relation_text = [
    'is the antonym of',
    'is at location of',
    'is capable of',
    'causes',
    'is created by',
    'is a kind of',
    'desires',
    'has subevent',
    'is part of',
    'has context',
    'has property',
    'is made of',
    'is not capable of',
    'does not desires',
    'is',
    'is related to',
    'is used for',
]
        rel_mapping = {
                        'CausesDesire': "causes desire",
                        'HasProperty': "has property",
                        'CapableOf': 'capable of',
                        'PartOf': 'part of',
                        'AtLocation': 'at location',
                        'Desires': 'desires',
                        'HasPrerequisite': 'has prerequisite',
                        'HasSubevent': 'has subevent',
                        'Antonym': 'antonym',
                        'Causes': 'causes',
        }
        #relation_text = rel_mapping.values()

        knowledge_embeddings = []
        # for rel in relation_text:
        #     # pdb.set_trace()
        #     rel_dict = self.roberta_tokenizer(rel, padding=True, max_length=7, truncation=True)
        #     rel_emb = self.roberta.embeddings(input_ids=torch.tensor(rel_dict['input_ids']).unsqueeze(0).to(self.roberta.device))
        #     knowledge_embeddings.append(rel_emb.mean(axis=1))
        # pdb.set_trace()
        knowledge_embeddings = None#torch.stack(knowledge_embeddings, axis=1)
        # pdb.set_trace()
        past_key_values = self.get_task_prompt(batch_size=batch_size * num_choices, knowledge_embeddings = knowledge_embeddings)
        #pdb.set_trace()
        if self.used_triplets_type == "individual":
            input_ids=combine_individual_view_triplets_input_ids
            token_type_ids=combine_individual_view_triplets_token_type_ids
            attention_mask=combine_individual_view_triplets_attention_mask
            position_ids=combine_individual_view_triplets_position_ids
        elif self.used_triplets_type == "selected_individual":
            input_ids=combine_selected_individual_view_triplets_input_ids
            token_type_ids=combine_selected_individual_view_triplets_token_type_ids
            attention_mask=combine_selected_individual_view_triplets_attention_mask
            position_ids=combine_selected_individual_view_triplets_position_ids
        elif self.used_triplets_type == "latent":
            input_ids=combine_latent_view_triplets_input_ids
            token_type_ids=combine_latent_view_triplets_token_type_ids
            attention_mask=combine_latent_view_triplets_attention_mask
            position_ids=combine_latent_view_triplets_position_ids
        elif self.used_triplets_type == "group":
            input_ids=combine_group_view_triplets_input_ids
            token_type_ids=combine_group_view_triplets_token_type_ids
            attention_mask=combine_group_view_triplets_attention_mask
            position_ids=combine_group_view_triplets_position_ids
        elif self.used_triplets_type == "retri":
            # print("****used_triplets_type == retri****")
            input_ids=combine_retri_view_triplets_input_ids
            token_type_ids=combine_retri_view_triplets_token_type_ids
            attention_mask=combine_retri_view_triplets_attention_mask
            position_ids=combine_retri_view_triplets_position_ids
        elif self.used_triplets_type == "meaning":
            # print("****used_triplets_type == retri****")
            input_ids=combine_meaning_view_triplets_input_ids
            token_type_ids=combine_meaning_view_triplets_token_type_ids
            attention_mask=combine_meaning_view_triplets_attention_mask
            position_ids=combine_meaning_view_triplets_position_ids
        elif self.used_triplets_type == "individual_group":
            # pdb.set_trace()
            input_ids=tv_individual_group_triplets_input_ids
            token_type_ids=tv_individual_group_triplets_token_type_ids
            attention_mask=tv_individual_group_triplets_attention_mask
            position_ids=tv_individual_group_triplets_position_ids
        elif self.used_triplets_type == "individual_retri":
            input_ids=tv_individual_retri_triplets_input_ids
            token_type_ids=tv_individual_retri_triplets_token_type_ids
            attention_mask=tv_individual_retri_triplets_attention_mask
            position_ids=tv_individual_retri_triplets_position_ids
        elif self.used_triplets_type == "individual_meaning":
            input_ids=tv_individual_meaning_triplets_input_ids
            token_type_ids=tv_individual_meaning_triplets_token_type_ids
            attention_mask=tv_individual_meaning_triplets_attention_mask
            position_ids=tv_individual_meaning_triplets_position_ids
        elif self.used_triplets_type == "individual_latent":
            input_ids=tv_individual_group_triplets_input_ids
            token_type_ids=tv_individual_latent_triplets_token_type_ids
            attention_mask=tv_individual_latent_triplets_attention_mask
            position_ids=tv_individual_latent_triplets_position_ids
        elif self.used_triplets_type == "group_individual":
            input_ids=tv_group_individual_triplets_input_ids
            token_type_ids=tv_group_individual_triplets_token_type_ids
            attention_mask=tv_group_individual_triplets_attention_mask
            position_ids=tv_group_individual_triplets_position_ids
        elif self.used_triplets_type == "all":
            input_ids=v3_individual_group_triplets_input_ids
            token_type_ids=v3_individual_group_triplets_token_type_ids
            attention_mask=v3_individual_group_triplets_attention_mask
            position_ids=v3_individual_group_triplets_position_ids
        elif self.used_triplets_type == "individual_retri_meaning":
            input_ids=v3_individual_retri_meaning_triplets_input_ids
            token_type_ids=v3_individual_retri_meaning_triplets_token_type_ids
            attention_mask=v3_individual_retri_meaning_triplets_attention_mask
            position_ids=v3_individual_retri_meaning_triplets_position_ids


        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        prefix_attention_mask = torch.ones(batch_size * num_choices, past_key_values[0].size(3)).to(self.albert.device)
        flat_attention_mask = torch.cat((prefix_attention_mask, flat_attention_mask), dim=1)
        outputs = self.albert(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        # outputs = self.roberta(
        #     flat_input_ids,
        #     position_ids=flat_position_ids,
        #     token_type_ids=flat_token_type_ids,
        #     attention_mask=flat_attention_mask,
        #     head_mask=head_mask,
        #     inputs_embeds=flat_inputs_embeds,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict
        # )
        pooled_output = self.dropout(outputs[1])
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class DebertaV2PrefixForMultipleChoice(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.classifier = torch.nn.Linear(output_dim, 1)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.init_weights()

        for param in self.deberta.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        deberta_param = 0
        for name, param in self.deberta.named_parameters():
            deberta_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - deberta_param
        print('total param is {}'.format(total_param))
    
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.deberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, num_choices = input_ids.shape[:2] if input_ids is not None else inputs_embeds.shape[:2]
        
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        past_key_values = self.get_prompt(batch_size=batch_size * num_choices)
        prefix_attention_mask = torch.ones(batch_size * num_choices, self.pre_seq_len).to(self.deberta.device)
        flat_attention_mask = torch.cat((prefix_attention_mask, flat_attention_mask), dim=1)

        outputs = self.deberta(
            flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            position_ids=flat_position_ids,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        encoder_layer = outputs[0]

        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DebertaPrefixForMultipleChoice(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.deberta = DebertaModel(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.classifier = torch.nn.Linear(output_dim, 1)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.init_weights()

        for param in self.deberta.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        deberta_param = 0
        for name, param in self.deberta.named_parameters():
            deberta_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - deberta_param
        print('total param is {}'.format(total_param))
    
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.deberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, num_choices = input_ids.shape[:2] if input_ids is not None else inputs_embeds.shape[:2]
        
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        past_key_values = self.get_prompt(batch_size=batch_size * num_choices)
        prefix_attention_mask = torch.ones(batch_size * num_choices, self.pre_seq_len).to(self.deberta.device)
        flat_attention_mask = torch.cat((prefix_attention_mask, flat_attention_mask), dim=1)

        outputs = self.deberta(
            flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            position_ids=flat_position_ids,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        encoder_layer = outputs[0]

        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class MultiExplDebertaPrefixForMultipleChoice(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.deberta = DebertaModel(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.classifier = torch.nn.Linear(output_dim, 1)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.init_weights()

        for param in self.deberta.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        deberta_param = 0
        for name, param in self.deberta.named_parameters():
            deberta_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - deberta_param
        print('total param is {}'.format(total_param))
    
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.deberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, num_choices = input_ids.shape[:2] if input_ids is not None else inputs_embeds.shape[:2]
        
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        past_key_values = self.get_prompt(batch_size=batch_size * num_choices)
        prefix_attention_mask = torch.ones(batch_size * num_choices, self.pre_seq_len).to(self.deberta.device)
        flat_attention_mask = torch.cat((prefix_attention_mask, flat_attention_mask), dim=1)

        outputs = self.deberta(
            flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            position_ids=flat_position_ids,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        encoder_layer = outputs[0]

        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertPromptForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        self.embeddings = self.bert.embeddings
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)

        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)

        bert_param = 0
        for name, param in self.bert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param)) # 9860105
    
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, num_choices = input_ids.shape[:2] if input_ids is not None else inputs_embeds[:2]

        input_ids = input_ids.reshape(-1, input_ids.size(-1)) if input_ids is not None else None
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        attention_mask = attention_mask.reshape(-1, attention_mask.size(-1)) if attention_mask is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        prompts = self.get_prompt(batch_size=batch_size * num_choices)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)

        prefix_attention_mask = torch.ones(batch_size * num_choices, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.bert(
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaPromptForMultipleChoice(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.embeddings = self.roberta.embeddings
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)

        self.init_weights()


        for param in self.roberta.parameters():
            param.requires_grad = False
            
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)

        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param))

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, num_choices = input_ids.shape[:2] if input_ids is not None else inputs_embeds.shape[:2]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        prompts = self.get_prompt(batch_size=batch_size * num_choices)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        prefix_attention_mask = torch.ones(batch_size * num_choices, self.pre_seq_len).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )