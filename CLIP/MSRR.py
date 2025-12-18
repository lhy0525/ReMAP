from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _infer_hw_from_tokens(n_tokens: int) -> Tuple[int, int]:
    h = int(math.sqrt(n_tokens))
    if h * h == n_tokens:
        return h, h
    # fallback to degenerate 1D grid
    return n_tokens, 1

# 3x3 的深度卷积模块
class DepthwiseConv3x3(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw(x)


class MSRRBlock(nn.Module):
    def __init__(
        self,
        d: int = 768,
        r: int = 3,               # reduced dim (paper uses r=3)
        n_experts: int = 4,       # i = 1..4 by default
        topk: Optional[int] = None,  # if None -> use all; else keep top-k in gating
        scales: Optional[list] = None,  # overrides default scales if provided
        upsample_mode: str = "bilinear",
    ):
        super().__init__()
        assert r >= 1, "r must be >= 1"
        self.d = d
        self.r = r
        self.n_experts = n_experts
        self.topk = topk if topk is not None else n_experts
        # scales s_i = 1 / 2^{i-1} for i=1..n
        if scales is None:
            self.scales = [1.0 / (2 ** (i)) if i > 0 else 1.0 for i in range(n_experts)]
            # i=0 -> 1.0, i=1 -> 0.5, i=2 -> 0.25, i=3 -> 0.125
        else:
            self.scales = scales
        self.upsample_mode = upsample_mode

        # Shared linear projections d->r and r->d
        self.proj_down = nn.Linear(d, r, bias=True)
        self.proj_up = nn.Linear(r, d, bias=True)

        # Gating parameters (Eq. 12-14): Wg, Wnoise in R^{r x n}
        self.Wg = nn.Parameter(torch.randn(r, n_experts) * 0.02)
        self.Wnoise = nn.Parameter(torch.randn(r, n_experts) * 0.02)

        # Expert depthwise conv after scaling
        self.dwconv = DepthwiseConv3x3(r)

        # Init
        nn.init.xavier_uniform_(self.proj_down.weight)
        nn.init.zeros_(self.proj_down.bias)
        nn.init.xavier_uniform_(self.proj_up.weight)
        nn.init.zeros_(self.proj_up.bias)

    def _gating_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, d)
        Return G(x): (B, n_experts) after KeepTopK+Softmax
        Implements Eq. (12)-(14)
        """
        B, N, d = x.shape
        # Eq.12: x_a = Reshape(AvgPool(Linear_d->r(x)), (B, r))
        xr = self.proj_down(x)                      # (B, N, r)
        xa = xr.mean(dim=1)                         # global avg over tokens -> (B, r)

        # Eq.13: H(x) = (xa Wg) + StandardNormal() * Softplus(xa Wnoise)
        Hg = xa @ self.Wg                           # (B, n)
        Hnoise_scale = F.softplus(xa @ self.Wnoise) # (B, n), >=0
        noise = torch.randn_like(Hg)
        H = Hg + noise * Hnoise_scale

        # KeepTopK then softmax
        if self.topk < self.n_experts:
            # mask others to -inf
            topk_vals, topk_idx = torch.topk(H, k=self.topk, dim=-1)
            mask = torch.full_like(H, float('-inf'))
            mask.scatter_(dim=-1, index=topk_idx, src=topk_vals)
            H = mask
        G = torch.softmax(H, dim=-1)                # (B, n)
        return G

    def _expert_op(self, xr: torch.Tensor, H: int, W: int, scale: float) -> torch.Tensor:
        """
        Single expert: Interpolate -> DWConv3x3 -> UpSample back to HxW
        xr: (B, N, r) reduced tokens
        Return: (B, N, r)
        """
        B, N, r = xr.shape
        feat = xr.view(B, H, W, r).permute(0, 3, 1, 2).contiguous()  # (B,r,H,W)
        # Interpolate to scaled size
        h_s = max(1, int(round(H * scale)))
        w_s = max(1, int(round(W * scale)))
        if (h_s, w_s) != (H, W):
            feat_s = F.interpolate(feat, size=(h_s, w_s), mode=self.upsample_mode, align_corners=False)
        else:
            feat_s = feat
        # DWConv 3x3
        feat_s = self.dwconv(feat_s)
        # UpSample back
        if (h_s, w_s) != (H, W):
            feat_o = F.interpolate(feat_s, size=(H, W), mode=self.upsample_mode, align_corners=False)
        else:
            feat_o = feat_s
        out = feat_o.permute(0, 2, 3, 1).reshape(B, N, r)            # (B,N,r)
        return out

    def forward(self, x: torch.Tensor, hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, d)
            hw: optional (H, W). If None, inferred from N as square.
        Returns:
            y: (B, N, d)  where y = x + sum_j G_j * Linear(r->d)(E_j(x))
        """
        assert x.dim() == 3, "input must be (B,N,d)"
        B, N, d = x.shape
        H, W = hw if hw is not None else _infer_hw_from_tokens(N)

        # Gating (on original x as specified)
        G = self._gating_scores(x)                  # (B, n)

        # Shared reduction for all experts before scaling
        xr = self.proj_down(x)                      # (B, N, r)

        # Experts: compute all E_j(x) in r-dim tokens
        expert_outs = []
        for s in self.scales[:self.n_experts]:
            e = self._expert_op(xr, H, W, s)       # (B, N, r)
            expert_outs.append(e)
        # Stack experts -> (B, n, N, r)
        E = torch.stack(expert_outs, dim=1)

        # Weight by gating G (B,n) → (B,n,1,1)
        Gw = G.view(B, self.n_experts, 1, 1)
        mixed_r = (Gw * E).sum(dim=1)               # (B, N, r)

        # Project back to d and residual
        y = self.proj_up(mixed_r)                   # (B, N, d)
        out = x + y                                  # residual
        return out
