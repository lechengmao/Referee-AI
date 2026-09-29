"""Train a matched VARS baseline from frozen five-window MViT caches."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.feature_cache import CachedFeatureDataset, collate_cached_features
from src.models.cached_vars_model import CachedVARSModel
from src.training.losses import class_frequency_weights
from src.training.trainer import fit
from src.utils.runtime import load_config, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--valid-cache', required=True)
    parser.add_argument('--config', default='configs/cached_vars_matched.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config['seed'])
    device = args.device if args.device != 'auto' else (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu')
    train = CachedFeatureDataset(args.train_cache)
    valid = CachedFeatureDataset(args.valid_cache)
    settings = config['training']
    train_loader = DataLoader(
        train, batch_size=settings['batch_size'], shuffle=True,
        collate_fn=collate_cached_features,
        generator=torch.Generator().manual_seed(config['seed']))
    valid_loader = DataLoader(
        valid, batch_size=settings['eval_batch_size'],
        collate_fn=collate_cached_features)
    weights = None
    if settings.get('class_weighted', False):
        weights = class_frequency_weights(
            train.records, device,
            settings.get('class_weight_mode', 'sqrt_inverse_frequency'))
    best = fit(CachedVARSModel(**config['model']), train_loader, valid_loader,
               config, Path(args.output), device, weights,
               config.get('loss', {}))
    print('Best checkpoint:', best)


if __name__ == '__main__':
    main()
