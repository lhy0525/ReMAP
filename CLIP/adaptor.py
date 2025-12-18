import torch
import torch.nn as nn
from .SGRP import SGRP 
from .MSRR import MSRRBlock 
from typing import Optional, List, Tuple

class Adapter(nn.Module):

    def __init__(self,
                 blocks: nn.ModuleList,  # Transformer blocks list
                 d: int,  # Embedding dimension
                 num_classes: int = 2,  # Number of classes (for text embeddings)
                 use_layers: Optional[List[int]] = None,  # Specify layers to use 
                 msrr_r: int = 3,  #  reduced dimension
                 msrr_topk: Optional[int] = None,  # Top-K gating for MSRR
                 temperature: float = 1.0):  # Temperature for SGRP
        super().__init__()
        
        self.blocks = blocks
        self.L = len(blocks)
        self.d = d
        self.num_classes = num_classes

        if use_layers is None:
            self.use_layers = set(range(self.L - 1))  # Use all layers except the last one
        else:
            self.use_layers = set([i for i in use_layers if 0 <= i < self.L - 1])

        self.msrr_list = nn.ModuleList([
            MSRRBlock(d=d, r=msrr_r, n_experts=4, topk=msrr_topk)
            if i in self.use_layers else nn.Identity()
            for i in range(self.L)
        ])
        self.sgrp_list = nn.ModuleList([
            SGRP(
                d=d,
                # init_N 可选；不清楚固定 patch 数时可以保持 None 走 lazy init
                init_N=None,
                l2_norm=True,
                temperature=temperature,
            ) if i in self.use_layers else nn.Identity()
            for i in range(self.L)
        ])

    def _split_cls_and_patches(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        g = x[:, 0, :]  # [CLS] token
        p = x[:, 1:, :]  # Patch tokens
        return g, p

    def _merge_cls_and_patches(self, g: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.cat([g.unsqueeze(1), p], dim=1)

    def forward(self, x: torch.Tensor, class_embeds: torch.Tensor, hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:

        B, NT, d = x.shape
        N = NT - 1  # Number of patches (excluding [CLS])

        for i, block in enumerate(self.blocks):
            if i in self.use_layers:
                # 1) Split CLS and patches
                g_i, p_i = self._split_cls_and_patches(x)

                # 2) Apply MSRR to update the patch tokens
                p_i = self.msrr_list[i](p_i, hw)

                m_bar_i = self.sgrp_list[i](p_i, class_embeds, g_i)

                # 5) Concatenate the [CLS] token, patches, and the generated prompts
                v_i = self._merge_cls_and_patches(g_i, p_i)
                x_in = torch.cat([v_i, m_bar_i], dim=1)

                # 6) Pass through the transformer block
                # ResidualAttentionBlock 期待 [L, N, D]，这里做 NLD->LND->NLD
                x_out = block(x_in.permute(1, 0, 2))
                x_out = x_out.permute(1, 0, 2)

                # 7) Keep only the visual tokens (exclude prompt tokens)
                x = x_out[:, : (N + 1), :]
            else:
                x = block(x.permute(1, 0, 2)).permute(1, 0, 2)

        return x
