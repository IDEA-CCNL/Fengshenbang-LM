# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""

from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from fengshen.models.groundedblip.groundingdino.utils import NestedTensor, clean_state_dict, is_main_process

from fengshen.models.groundedblip.backbone.swin_transformer import SwinTransformer
from fengshen.models.groundedblip.backbone.position_encoding import PositionEmbeddingSineHW

class Joiner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.swintransformer = SwinTransformer(args)
        self.position_embedding = PositionEmbeddingSineHW(args.hidden_dim,
                                                          temperatureh=args.pe_temperatureh,
                                                          temperaturew=args.pe_temperaturew,
                                                          normalize=True,
                                                          )
        bb_num_channels = self.swintransformer.num_features[4 - len(tuple(args.return_interm_indices)) :]
        self.num_channels = bb_num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.swintransformer(tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self.position_embedding(x).to(x.tensors.dtype))

        return out, pos
