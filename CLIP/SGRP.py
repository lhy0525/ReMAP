from __future__ import annotations
from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """对嵌入向量做 L2 归一化，避免量纲差与数值爆炸。"""
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))


class SGRP(nn.Module):
    def __init__(
        self,
        d: int,
        init_N: Optional[int] = None,
        l2_norm: bool = True,
        pixel_norm: Literal["none", "softmax", "l1"] = "none",
        img_class_norm: Literal["none", "softmax", "l2"] = "none",
        temperature: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.d = d
        self.init_N = init_N

    
        self.l2_norm = l2_norm
        self.pixel_norm = pixel_norm
        self.img_class_norm = img_class_norm
        self.temperature = float(temperature)
        self.eps = eps


        self.h: Optional[nn.Parameter] = None
        if init_N is not None:
            self.h = nn.Parameter(torch.randn(init_N, d) * 0.02)

    def _ensure_h(self, N: int, device: torch.device, dtype: torch.dtype):

        if self.h is None:
            self.h = nn.Parameter(
                torch.randn(N, self.d, device=device, dtype=dtype) * 0.02,
                requires_grad=True,
            )
        elif self.h.shape[0] != N:
            with torch.no_grad():
                oldN, d = self.h.shape
                if N > oldN:
                    pad = torch.randn(N - oldN, d, device=self.h.device, dtype=self.h.dtype) * 0.02
                    self.h = nn.Parameter(torch.cat([self.h.data, pad], dim=0), requires_grad=True)
                else:
                    self.h = nn.Parameter(self.h.data[:N], requires_grad=True)

    def forward(self, p: torch.Tensor, t: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p: [B, N, d]     patch tokens
            t: [C, d] 或 [B, C, d]  类别嵌入 (class_embeds)
            g: [B, d] 或 [B, 1, d]  图像级 [CLS] token

        Returns:
            m_bar: [B, C, d]  每类一个 d 维视觉提示 (prompt)
        """
        assert p.dim() == 3, f"p shape should be [B,N,d], got {p.shape}"
        B, N, d = p.shape

        # ---- 统一 t 形状到 [B, C, d] ----
        if t.dim() == 2:  # [C, d]
            C = t.size(0)
            t = t.unsqueeze(0).expand(B, -1, -1)       # [B, C, d]
        elif t.dim() == 3:  # [B, C, d]
            assert t.size(0) == B, "batch size of t must match p"
            C = t.size(1)
        else:
            raise ValueError(f"t shape should be [C,d] or [B,C,d], got {t.shape}")

        # ---- 统一 g 形状到 [B, C, d]（在类维广播）----
        if g.dim() == 2:       # [B, d]
            g = g.unsqueeze(1)  # [B, 1, d]
        elif g.dim() == 3:     # [B, 1, d]
            assert g.size(1) == 1, "g second dim should be 1 (i.e., [B,1,d])"
        else:
            raise ValueError(f"g shape should be [B,d] or [B,1,d], got {g.shape}")
        g = g.expand(-1, C, -1)  # [B, C, d]

        if self.l2_norm:
            p = l2_normalize(p, dim=-1, eps=self.eps)
            t = l2_normalize(t, dim=-1, eps=self.eps)
            g = l2_normalize(g, dim=-1, eps=self.eps)

        
        Tg = t * g  # [B, C, d]

        if self.img_class_norm == "softmax":
            Tg = torch.softmax(Tg / max(self.temperature, 1e-8), dim=1)
        elif self.img_class_norm == "l2":
            Tg = l2_normalize(Tg, dim=1, eps=self.eps)

        # ---- 像素级加权 m = p @ Tg^T ----
        # p:  [B, N, d]
        # Tg: [B, C, d] -> [B, d, C]
        m = torch.matmul(p, Tg.transpose(1, 2))  # [B, N, C]

        if self.pixel_norm == "softmax":
            m = torch.softmax(m / max(self.temperature, 1e-8), dim=1)  # across N
        elif self.pixel_norm == "l1":
            denom = m.abs().sum(dim=1, keepdim=True).clamp_min(self.eps)
            m = m / denom
            
        # m: [B, N, C] -> [B, C, N]
        self._ensure_h(N, device=p.device, dtype=p.dtype)
        m_T = m.transpose(1, 2)  # [B, C, N]
        # [B, C, d] = [B, C, N] @ [N, d]
        m_bar = torch.matmul(m_T, self.h)

        return m_bar  # [B, C, d]
