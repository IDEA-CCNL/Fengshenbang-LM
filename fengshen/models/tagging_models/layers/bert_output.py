import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class TokenClassifierOutput:
    """
    Base class for outputs of token classification models.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


@dataclass
class SpanClassifierOutput:
    """
    Base class for outputs of span classification models.
    """
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.LongTensor = None
    end_logits: torch.LongTensor = None


@dataclass
class BiaffineClassifierOutput:
    """
    Base class for outputs of span classification models.
    """
    loss: Optional[torch.FloatTensor] = None
    span_logits: torch.FloatTensor = None
