"""Original VARS cross-view fusion over temporally averaged cached features."""
import torch
from torch import nn

from .classifier import ClassificationHeads
from .vars_attention import VARSAttention


class CachedVARSModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=768,
                 aggregate_projected=False, reference_style=False):
        super().__init__()
        self.aggregation = VARSAttention(input_dim, aggregate_projected)
        self.heads = ClassificationHeads(input_dim, hidden_dim, reference_style)

    def forward(self, features, mask):
        if features.ndim != 4 or mask.shape != features.shape[:3]:
            raise ValueError('Expected [B,V,T,D] features and [B,V,T] mask')
        counts = mask.sum(2).clamp_min(1)
        per_view = (features * mask[..., None]).sum(2) / counts[..., None]
        view_mask = mask.any(2)
        pooled, view_attention = self.aggregation(per_view, view_mask)
        # Expose a compatible spatiotemporal map. VARS has no temporal selector,
        # so each valid window receives an equal share of its view's weight.
        attention = (view_attention[..., None] / counts[..., None]).masked_fill(~mask, 0)
        return {**self.heads(pooled), 'attention': attention}
