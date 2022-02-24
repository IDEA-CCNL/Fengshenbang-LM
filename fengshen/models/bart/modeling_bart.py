import warnings
from pytorch_lightning import LightningModule
from fengshen.models import transformer_utils

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, Tuple

from transformers.file_utils import add_end_docstrings, replace_return_docstrings
from transformers.modeling_outputs import ModelOutput, Seq2SeqLMOutput
from transformers.models.bart import BartPretrainedModel, BartConfig, BartModel
from transformers.models.bart.modeling_bart import BartClassificationHead


_CONFIG_FOR_DOC = "BartConfig"


# ------------------------ ZZ: CBart addition ------------------------


def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


BART_GENERATION_EXAMPLE = r"""
    Summarization example::

        >>> from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

        >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

        >>> ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
        >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

        >>> # Generate Summary
        >>> summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        >>> print([tokenizer.decode(g, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False) for g in summary_ids])

    Mask filling example::

        >>> from transformers import BartTokenizer, BartForConditionalGeneration
        >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        >>> TXT = "My friends are <mask> but they eat too many carbs."

        >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
        >>> logits = model(input_ids).logits

        >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
        >>> probs = logits[0, masked_index].softmax(dim=0)
        >>> values, predictions = probs.topk(5)

        >>> tokenizer.decode(predictions).split()
"""


@dataclass
class CBartLMOutput(ModelOutput):
    """
    Base class for CBart specific language models outputs.

    Args:
        ....
    """
    loss: Optional[torch.FloatTensor] = None
    encoder_loss: Optional[torch.FloatTensor] = None
    decoder_loss: Optional[torch.FloatTensor] = None
    encoder_logits: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class BartForTextInfill(BartPretrainedModel):
    """
    this class is designed for text infilling.
    During training, the encoder is used to predict replace, insert,
    and the decoder is used to generate original input.
    Compared with BartForConditionalGeneration class,
    we add a module over the encoder and add a new loss for the encoder.
    """
    base_model_prefix = "model"
    authorized_missing_keys = [r"final_logits_bias",
                               r"encoder\.version", r"decoder\.version"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = BartModel(config)
        self.model = base_model
        self.register_buffer("final_logits_bias", torch.zeros(
            (1, self.model.shared.num_embeddings)))
        # print( config.encoder_loss_type, config.num_labels)

        # add a new attribute into BartConfig class (revise BartConfig)
        self.encoder_loss_type = config.encoder_loss_type
        self.num_labels = config.num_labels
        if self.encoder_loss_type == 0:  # 0 is classification loss, 1 is regression loss
            # add a classification module for the encoder
            self.classification_head = BartClassificationHead(
                config.d_model, config.d_model, config.num_labels, config.classif_dropout,
            )
        else:
            # add a regression module for the encoder
            self.classification_head = BartClassificationHead(
                config.d_model, config.d_model, 1, config.classif_dropout,
            )

        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)
        self.loss_weight = config.loss_weight
        self.register_buffer("label_weights", torch.zeros((self.num_labels)))

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        old_num_tokens = self.model.shared.num_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.model.shared = new_embeddings
        self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens),
                                     device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        encoder_labels=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        **unused,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`,
                `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should either be in ``[0, ..., config.vocab_size]``
                or -100 (see ``input_ids`` docstring).
            Tokens with indices set to ``-100`` are ignored (masked),
                the loss is only computed for the tokens
            with labels in ``[0, ..., config.vocab_size]``.

    Returns:

    Conditional generation example::

            # Mask filling only works for bart-large
            from transformers import BartTokenizer, BartForConditionalGeneration
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            TXT = "My friends are <mask> but they eat too many carbs."

            model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
            input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
            logits = model(input_ids).logits

            masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            probs = logits[0, masked_index].softmax(dim=0)
            values, predictions = probs.topk(5)

            tokenizer.decode(predictions).split()
            # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, "
                + "use `decoder_past_key_values` instead.",
                FutureWarning,
            )
            unused.pop("decoder_cached_states")
        return_dict = return_dict if return_dict is not None else False

        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # logits and loss for the encoder
        # last hidden state
        encoder_last_hidden_state = outputs['encoder_last_hidden_state']
        # eos_mask = input_ids.eq(self.config.eos_token_id)
        # if len(torch.unique(eos_mask.sum(1))) > 1:
        #     raise ValueError("All examples must have the same number of <eos> tokens.")
        # sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
        encoder_logits = self.classification_head(encoder_last_hidden_state)
        encoder_loss = None
        if encoder_labels is not None:
            # classification loss
            if self.encoder_loss_type == 0:
                # ZZ: seems like MSE loss does not support weighting, so only CEL has weighting applied for now
                loss_fct = nn.CrossEntropyLoss(weight=self.label_weights)
                encoder_loss = loss_fct(
                    encoder_logits.view(-1, self.config.num_labels), encoder_labels.view(-1))
            # regression loss
            else:
                encoder_logits = encoder_logits.view(
                    encoder_logits.size(0), -1)
                encoder_logits = torch.sigmoid(
                    encoder_logits) * self.num_labels - 0.5
                loss_fct = nn.MSELoss(reduction='none')
                _loss = loss_fct(encoder_logits, encoder_labels)
                encoder_loss = torch.mean(_loss[encoder_labels >= 0])
                # encoder_loss =_loss[encoder_labels>=0]

        # logits and loss for the decoder
        lm_logits = F.linear(
            outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # TODO(SS): do we need to ignore pad tokens in labels?
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        loss = None
        if masked_lm_loss is not None and encoder_loss is not None:
            loss = encoder_loss * self.loss_weight + masked_lm_loss

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CBartLMOutput(
            loss=loss,
            encoder_loss=encoder_loss,
            decoder_loss=masked_lm_loss,
            encoder_logits=encoder_logits,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, use_cache, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        encoder_outputs, past_key_values = past
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            # change this to avoid caching (presumably for debugging)
            "use_cache": use_cache,
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1:
            self._force_token_ids_generation(logits, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_ids) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        all_but_token_ids_mask = torch.tensor(
            [x for x in range(self.config.vocab_size) if x not in token_ids],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        assert len(
            scores.shape) == 2, "scores should be of rank 2 with shape: [batch_size, vocab_size]"
        scores[:, all_but_token_ids_mask] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        ((enc_out, enc_mask), past_key_values) = past
        reordered_past = []
        for layer_past in past_key_values:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        new_enc_out = enc_out if enc_out is None else enc_out.index_select(
            0, beam_idx)
        new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(
            0, beam_idx)

        past = ((new_enc_out, new_enc_mask), reordered_past)
        return past

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.model.shared)  # make it on the fly

    def get_encoder_logits(self, input_ids, attention_mask=None):
        # print(input_ids, attention_mask)
        # encoder_outputs = self.model.get_encoder_outputs(
        #         self,
        #         input_ids,
        #         attention_mask=attention_mask,
        #         output_attentions=None,
        #         output_hidden_states=None,
        #         return_dict=None,
        #  )

        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # logits and loss for the encoder
        # last hidden state
        encoder_last_hidden_state = encoder_outputs['last_hidden_state']
        encoder_logits = self.classification_head(encoder_last_hidden_state)

        # classification
        if self.encoder_loss_type == 0:
            # probs = torch.softmax(encoder_logits,dim=-1)
            pass
        # regression
        else:
            encoder_logits = encoder_logits.view(encoder_logits.size(0), -1)
            encoder_logits = torch.sigmoid(
                encoder_logits) * self.num_labels - 0.5
        return encoder_outputs, encoder_logits


class CBartLightning(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_args):
        parser = parent_args.add_argument_group("CBart specific parameters")
        parser.add_argument('--num_labels', type=int, default=3)
        parser.add_argument('--encoder_loss_type', type=int, default=0)
        parser.add_argument('--loss_weight', type=float, default=1.0)
        parser.add_argument('--label_weights', type=float, nargs='+', default=[1.0, 1.0, 1.0])
        parser.add_argument('--masked_lm', type=float, default=0)
        return parent_args

    def __init__(
        self,
        args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = BartForTextInfill.from_pretrained(args.model_path, num_labels=self.hparams.num_labels,
                                                       encoder_loss_type=self.hparams.encoder_loss_type,
                                                       loss_weight=self.hparams.loss_weight,)
        self.model.label_weights = torch.tensor(
            self.hparams.label_weights, dtype=torch.half)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        return outputs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss = outputs["loss"]

        return {"loss": val_loss}

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()

        # Calculate total steps
        tb_size = self.hparams.train_batchsize * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        transformer_utils.configure_optimizers(self)
