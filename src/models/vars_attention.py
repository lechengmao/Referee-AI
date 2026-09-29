"""VARS equations (3)-(6), with padding and a zero-denominator safeguard."""
import torch
from torch import nn


class VARSAttention(nn.Module):
    def __init__(self, dim, aggregate_projected=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        nn.init.uniform_(self.weight, -1, 1)  # official code initialization
        self.aggregate_projected = aggregate_projected

    def forward(self, features, mask=None):
        if features.ndim != 3:
            raise ValueError('Expected B,V,D features')
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        if mask.dtype != torch.bool or mask.shape != features.shape[:2] or not mask.any(1).all():
            raise ValueError('Each incident needs at least one valid view and a B,V boolean mask')
        f = features.masked_fill(~mask[..., None], 0)
        projected = f @ self.weight
        scores = (projected @ projected.transpose(1, 2)).relu()
        scores = scores.masked_fill(~(mask[:, :, None] & mask[:, None, :]), 0)
        totals = scores.sum((1, 2), keepdim=True)
        normal = scores / totals.clamp_min(torch.finfo(scores.dtype).tiny)
        weights = normal.sum(-1)
        uniform = mask.to(f.dtype) / mask.sum(1, keepdim=True)
        weights = torch.where(totals.flatten(1) > 0, weights, uniform)
        values = projected if self.aggregate_projected else f
        return (values * weights[..., None]).sum(1), weights
