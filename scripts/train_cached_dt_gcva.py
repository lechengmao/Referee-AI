"""Train spatiotemporal DT-GCVA from frozen MViT feature caches."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.feature_cache import CachedFeatureDataset, collate_cached_features
from src.models.spatiotemporal_model import SpatiotemporalDTGCVAModel
from src.training.trainer import fit
from src.training.losses import class_frequency_weights
from src.utils.runtime import load_config, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-cache', required=True)
    parser.add_argument('--valid-cache', required=True)
    parser.add_argument('--config', default='configs/spatiotemporal_dt_gcva.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config['seed'])
    device = args.device if args.device != 'auto' else (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu')
    train, valid = CachedFeatureDataset(args.train_cache), CachedFeatureDataset(args.valid_cache)
    settings = config['training']
    train_loader = DataLoader(train, batch_size=settings['batch_size'], shuffle=True,
                              collate_fn=collate_cached_features,
                              generator=torch.Generator().manual_seed(config['seed']))
    valid_loader = DataLoader(valid, batch_size=settings['eval_batch_size'],
                              collate_fn=collate_cached_features)
    class_weights = None
    if settings.get('class_weighted', False):
        class_weights = class_frequency_weights(
            train.records, device,
            settings.get('class_weight_mode', 'sqrt_inverse_frequency'))
    best = fit(SpatiotemporalDTGCVAModel(**config['model']), train_loader, valid_loader,
               config, Path(args.output), device, class_weights,
               config.get('loss', {}))
    print('Best checkpoint:', best)


if __name__ == '__main__':
    main()
