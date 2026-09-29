import torch
from torch import nn
from .mvit_encoder import MViTEncoder
from .vars_attention import VARSAttention
from .dt_gcva import DTGCVAFusion
from .classifier import ClassificationHeads


class VARSModel(nn.Module):
    def __init__(self, encoder=None, pretrained=True, representation='embedding',
                 hidden_dim=768, aggregate_projected=False, reference_style=False,
                 aggregation_type='vars', fusion_dim=256, fusion_heads=4,
                 fusion_dropout=0.1, view_dropout=0.2):
        super().__init__()
        self.encoder = encoder if encoder is not None else MViTEncoder(pretrained, representation)
        dim = self.encoder.output_dim
        if aggregation_type == 'vars':
            self.aggregation = VARSAttention(dim, aggregate_projected)
        elif aggregation_type == 'dt_gcva':
            self.aggregation = DTGCVAFusion(
                dim, fusion_dim, fusion_heads, fusion_dropout, view_dropout)
        else:
            raise ValueError(f'Unknown aggregation_type: {aggregation_type!r}')
        self.aggregation_type = aggregation_type
        self.heads = ClassificationHeads(dim, hidden_dim, reference_style)

    def forward(self, video, mask):
        if video.ndim != 6 or mask.dtype != torch.bool or mask.shape != video.shape[:2]:
            raise ValueError('Expected B,V,C,T,H,W video and B,V boolean mask')
        if not mask.any(1).all():
            raise ValueError('Each incident must have at least one valid view')
        # Never encode padded views. Indexed assignment preserves gradients.
        encoded = self.encoder(video[mask])
        features = encoded.new_zeros((*mask.shape, encoded.shape[-1]))
        features[mask] = encoded
        if self.aggregation_type == 'vars':
            pooled, attention = self.aggregation(features, mask)
            return {**self.heads(pooled), 'attention': attention}
        fusion = self.aggregation(features, mask)
        logits = self.heads.forward_task_features(
            fusion['foul_features'], fusion['severity_features'])
        return {**logits,
                'attention': fusion['attention'],
                'foul_attention': fusion['foul_attention'],
                'severity_attention': fusion['severity_attention']}
