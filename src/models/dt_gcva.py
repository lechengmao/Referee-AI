"""Dual-Task Gated Cross-View Attention (DT-GCVA)."""
import math
import torch
from torch import nn


class DTGCVAFusion(nn.Module):
    """Permutation-invariant fusion with task-specific attention pooling.

    A small self-attention block lets views exchange evidence. Two learned
    queries then pool different evidence for foul type and severity. A learned
    gate mixes each task representation with masked mean pooling as a stable
    residual path for the small MVFoul dataset.
    """

    def __init__(self, input_dim, fusion_dim=256, num_heads=4, dropout=0.1,
                 view_dropout=0.2, use_mean_residual=True,
                 pooling_temperature=None):
        super().__init__()
        if fusion_dim < 1 or num_heads < 1 or fusion_dim % num_heads:
            raise ValueError('fusion_dim must be positive and divisible by num_heads')
        if not 0 <= view_dropout < 1:
            raise ValueError('view_dropout must be in [0, 1)')
        self.view_dropout = view_dropout
        self.use_mean_residual = use_mean_residual
        if pooling_temperature is not None and pooling_temperature <= 0:
            raise ValueError('pooling_temperature must be positive or None')
        self.pooling_temperature = pooling_temperature
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.view_attention = (nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True)
            if pooling_temperature is None else None)
        self.view_norm = nn.LayerNorm(fusion_dim) if pooling_temperature is None else None
        self.task_queries = nn.Parameter(torch.empty(2, fusion_dim))
        nn.init.normal_(self.task_queries, std=0.02)
        self.task_attention = (nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True)
            if pooling_temperature is None else None)
        # sigmoid(0) starts both tasks at an equal attention/mean mixture.
        self.gates = nn.Parameter(torch.zeros(2, fusion_dim))
        self.output = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, input_dim))

    def _drop_views(self, mask):
        if not self.training or self.view_dropout == 0:
            return mask
        kept = mask & (torch.rand(mask.shape, device=mask.device) >= self.view_dropout)
        empty = ~kept.any(1)
        if empty.any():
            # Restore the first valid view, guaranteeing a usable set.
            first = mask.to(torch.int64).argmax(1)
            kept[empty, first[empty]] = True
        return kept

    def forward(self, features, mask=None):
        if features.ndim != 3:
            raise ValueError('Expected B,V,D features')
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        if mask.dtype != torch.bool or mask.shape != features.shape[:2] or not mask.any(1).all():
            raise ValueError('Each incident needs at least one valid view and a B,V boolean mask')
        active = self._drop_views(mask)
        local_tokens = self.project(features).masked_fill(~active[..., None], 0)
        if self.view_attention is None:
            tokens = local_tokens
        else:
            exchanged, _ = self.view_attention(
                local_tokens, local_tokens, local_tokens,
                key_padding_mask=~active, need_weights=False)
            tokens = self.view_norm(local_tokens + exchanged).masked_fill(~active[..., None], 0)

        queries = self.task_queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        if self.task_attention is None:
            # Score and pool local evidence so reported weights remain faithful
            # to an individual view/window instead of a globally mixed token.
            scores = torch.einsum('bqd,bvd->bqv', queries, local_tokens)
            scores = scores / (math.sqrt(local_tokens.shape[-1]) * self.pooling_temperature)
            scores = scores.masked_fill(~active[:, None, :], float('-inf'))
            weights = scores.softmax(-1)
            pooled = torch.einsum('bqv,bvd->bqd', weights, local_tokens)
        else:
            pooled, weights = self.task_attention(
                queries, tokens, tokens, key_padding_mask=~active,
                need_weights=True, average_attn_weights=True)
            weights = weights.masked_fill(~active[:, None, :], 0)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny)
        if self.use_mean_residual:
            mean = tokens.sum(1) / active.sum(1, keepdim=True)
            gate = self.gates.sigmoid().unsqueeze(0)
            fused = gate * pooled + (1 - gate) * mean.unsqueeze(1)
        else:
            fused = pooled
        fused = self.output(fused)
        return {
            'foul_features': fused[:, 0],
            'severity_features': fused[:, 1],
            'foul_attention': weights[:, 0],
            'severity_attention': weights[:, 1],
            'attention': weights.mean(1),
            'active_view_mask': active,
        }


class HierarchicalDTGCVAFusion(nn.Module):
    """Fuse views inside each window before selecting temporal evidence."""

    def __init__(self, input_dim, fusion_dim=256, dropout=0.1,
                 token_dropout=0.1, view_temperature=0.25,
                 temporal_temperature=0.20, use_mean_residual=True):
        super().__init__()
        for value in (view_temperature, temporal_temperature):
            if value <= 0:
                raise ValueError('Hierarchical temperatures must be positive')
        self.token_dropout = token_dropout
        self.view_temperature = view_temperature
        self.temporal_temperature = temporal_temperature
        self.use_mean_residual = use_mean_residual
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, fusion_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.view_queries = nn.Parameter(torch.empty(2, fusion_dim))
        self.temporal_queries = nn.Parameter(torch.empty(2, fusion_dim))
        nn.init.normal_(self.view_queries, std=0.02)
        nn.init.normal_(self.temporal_queries, std=0.02)
        self.gates = nn.Parameter(torch.zeros(2, fusion_dim))
        self.output = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, input_dim))

    def _drop_tokens(self, mask):
        if not self.training or self.token_dropout == 0:
            return mask
        kept = mask & (torch.rand(mask.shape, device=mask.device) >= self.token_dropout)
        # Every valid time window must retain at least one view.
        empty = mask.any(1) & ~kept.any(1)
        if empty.any():
            first = mask.to(torch.int64).argmax(1)
            b, t = empty.nonzero(as_tuple=True)
            kept[b, first[b, t], t] = True
        return kept

    def forward(self, features, mask):
        if features.ndim != 4 or mask.shape != features.shape[:3]:
            raise ValueError('Expected [B,V,T,D] features and [B,V,T] mask')
        active = self._drop_tokens(mask)
        local = self.project(features).masked_fill(~active[..., None], 0)
        scale = math.sqrt(local.shape[-1])

        # [B,Q,V,T], normalized only across cameras for each time window.
        view_scores = torch.einsum('qd,bvtd->bqvt', self.view_queries, local)
        view_scores = view_scores / (scale * self.view_temperature)
        view_scores = view_scores.masked_fill(~active[:, None], float('-inf'))
        valid_time = active.any(1)
        view_scores = view_scores.masked_fill(
            ~valid_time[:, None, None, :], 0)
        view_weights = view_scores.softmax(2).masked_fill(
            ~active[:, None], 0)
        view_weights = view_weights / view_weights.sum(2, keepdim=True).clamp_min(
            torch.finfo(view_weights.dtype).tiny)
        time_tokens = torch.einsum('bqvt,bvtd->bqtd', view_weights, local)

        # Each task selects among its five already multi-view-fused windows.
        temporal_scores = torch.einsum(
            'qd,bqtd->bqt', self.temporal_queries, time_tokens)
        temporal_scores = temporal_scores / (scale * self.temporal_temperature)
        temporal_scores = temporal_scores.masked_fill(
            ~valid_time[:, None], float('-inf'))
        temporal_weights = temporal_scores.softmax(-1)
        pooled = torch.einsum('bqt,bqtd->bqd', temporal_weights, time_tokens)

        if self.use_mean_residual:
            count = active.sum((1, 2), keepdim=True).clamp_min(1)
            mean = local.sum((1, 2)) / count.squeeze(-1)
            gate = self.gates.sigmoid().unsqueeze(0)
            pooled = gate * pooled + (1 - gate) * mean.unsqueeze(1)
        fused = self.output(pooled)
        spatiotemporal = view_weights * temporal_weights[:, :, None, :]
        return {
            'foul_features': fused[:, 0],
            'severity_features': fused[:, 1],
            'foul_attention': spatiotemporal[:, 0],
            'severity_attention': spatiotemporal[:, 1],
            'attention': spatiotemporal.mean(1),
            'active_view_mask': active,
        }
