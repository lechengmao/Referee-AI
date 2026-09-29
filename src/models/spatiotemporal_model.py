"""Classifier over cached view-by-time MViT embeddings."""
import torch
from torch import nn

from .classifier import ClassificationHeads
from .dt_gcva import DTGCVAFusion, HierarchicalDTGCVAFusion


class SpatiotemporalDTGCVAModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=768, fusion_dim=256,
                 fusion_heads=4, fusion_dropout=0.1, token_dropout=0.2,
                 attention_entropy_weight=0.05,
                 attention_entropy_floor_ratio=0.35,
                 pooling_temperature=0.25, use_mean_residual=True,
                 fusion_strategy='flat', view_temperature=0.25,
                 temporal_temperature=0.20):
        super().__init__()
        if fusion_strategy == 'flat':
            self.aggregation = DTGCVAFusion(
                input_dim, fusion_dim, fusion_heads, fusion_dropout,
                token_dropout, use_mean_residual=use_mean_residual,
                pooling_temperature=pooling_temperature)
        elif fusion_strategy == 'hierarchical':
            self.aggregation = HierarchicalDTGCVAFusion(
                input_dim, fusion_dim, fusion_dropout, token_dropout,
                view_temperature, temporal_temperature, use_mean_residual)
        else:
            raise ValueError(f'Unsupported fusion strategy: {fusion_strategy}')
        self.heads = ClassificationHeads(input_dim, hidden_dim)
        self.attention_entropy_weight = attention_entropy_weight
        if not 0 <= attention_entropy_floor_ratio <= 1:
            raise ValueError('attention_entropy_floor_ratio must be in [0, 1]')
        self.attention_entropy_floor_ratio = attention_entropy_floor_ratio

    def forward(self, features, mask):
        if features.ndim != 4 or mask.shape != features.shape[:3] or mask.dtype != torch.bool:
            raise ValueError('Expected [B,V,T,D] features and [B,V,T] boolean mask')
        batch, views, windows, dim = features.shape
        if isinstance(self.aggregation, HierarchicalDTGCVAFusion):
            fused = self.aggregation(features, mask)
        else:
            fused = self.aggregation(features.reshape(batch, views * windows, dim),
                                     mask.reshape(batch, views * windows))
        logits = self.heads.forward_task_features(
            fused['foul_features'], fused['severity_features'])
        foul = fused['foul_attention'].reshape(batch, views, windows)
        severity = fused['severity_attention'].reshape(batch, views, windows)
        foul_entropy = -(foul.clamp_min(1e-8).log() * foul).sum((1, 2))
        severity_entropy = -(severity.clamp_min(1e-8).log() * severity).sum((1, 2))
        max_entropy = mask.sum((1, 2)).float().log().clamp_min(1e-8)
        floor = self.attention_entropy_floor_ratio * max_entropy
        # Penalize only pathologically sharp attention. The previous positive
        # entropy term minimized entropy and therefore encouraged collapse.
        entropy_penalty = (
            torch.relu(floor - foul_entropy).square()
            + torch.relu(floor - severity_entropy).square()
        ).mean()
        return {**logits,
                'attention': fused['attention'].reshape(batch, views, windows),
                'foul_attention': foul,
                'severity_attention': severity,
                'foul_view_attention': foul.sum(2),
                'severity_view_attention': severity.sum(2),
                'foul_temporal_attention': foul.sum(1),
                'severity_temporal_attention': severity.sum(1),
                'foul_attention_entropy': foul_entropy,
                'severity_attention_entropy': severity_entropy,
                'auxiliary_loss': self.attention_entropy_weight * entropy_penalty}
