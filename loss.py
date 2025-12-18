import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from math import exp
    
def dice_loss(inputs, targets, num_masks, eps: float = 1.0, from_logits=False):
    """
    inputs: logits, shape [B, 1, H, W] 或 [B, C, H, W]
    targets: 概率/0-1 掩码, shape 与 inputs 匹配（通道数相同或为 1）
    """
    if from_logits:
        inputs = inputs.sigmoid()
    # 与 targets 对齐形状
    inputs = inputs.float().flatten(1)   # [B, HW] 或 [B, C*HW]
    targets = targets.float().flatten(1) # [B, HW] 或 [B, C*HW]
    numerator   = 2.0 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1.0 - (numerator + eps) / (denominator + eps)
    return loss.sum() / float(num_masks)
def focal_loss(inputs, targets, alpha=-1, gamma=4, reduction="mean"):
    inputs = inputs.float()
    targets = targets.float()
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss
