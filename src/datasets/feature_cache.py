"""Datasets and batching for deterministic cached view-by-time embeddings."""
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    def __init__(self, path):
        self.path = Path(path)
        payload = torch.load(self.path, map_location='cpu', weights_only=True)
        if payload.get('format_version') != 1 or not payload.get('records'):
            raise ValueError(f'Unsupported or empty feature cache: {self.path}')
        self.metadata = payload['metadata']
        self.records = payload['records']
        for record in self.records:
            features = record['features']
            if features.ndim != 3 or features.shape[-1] < 1:
                raise ValueError('Cached features must have shape [views,windows,dim]')
            if not torch.isfinite(features).all():
                raise ValueError('Cached features contain non-finite values')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def collate_cached_features(items):
    if not items:
        raise ValueError('Empty batch')
    dim = items[0]['features'].shape[-1]
    max_views = max(x['features'].shape[0] for x in items)
    max_windows = max(x['features'].shape[1] for x in items)
    features = items[0]['features'].new_zeros((len(items), max_views, max_windows, dim))
    mask = torch.zeros(len(items), max_views, max_windows, dtype=torch.bool)
    for batch_index, item in enumerate(items):
        value = item['features']
        if value.ndim != 3 or value.shape[-1] != dim:
            raise ValueError('Inconsistent cached feature shape')
        views, windows = value.shape[:2]
        features[batch_index, :views, :windows] = value
        mask[batch_index, :views, :windows] = True
    return {'features': features, 'mask': mask,
            'labels': torch.tensor([x['labels'] for x in items], dtype=torch.long),
            'ids': [x['id'] for x in items], 'paths': [x['paths'] for x in items],
            'window_starts': [x['window_starts'] for x in items]}
