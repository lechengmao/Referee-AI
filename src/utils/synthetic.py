"""Test-only encoder/data: never an alternative research backbone or dataset."""
import torch
from torch import nn
from torch.utils.data import Dataset


class TinyTestEncoder(nn.Module):
    output_dim = 12

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(3, 12, 1), nn.ReLU(), nn.AdaptiveAvgPool3d(1))

    def forward(self, video):
        return self.net(video).flatten(1)


class SyntheticIncidents(Dataset):
    def __init__(self, count=8, seed=123):
        self.count, self.seed = count, seed

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        views = 1 + i % 3
        video = torch.randn(views, 3, 4, 8, 8, generator=torch.Generator().manual_seed(self.seed+i))
        return {'video': video, 'labels': (i % 8, i % 4),
                'id': f'SYNTHETIC/{self.seed}/{i}', 'paths': [f'synthetic/view_{v}' for v in range(views)]}
