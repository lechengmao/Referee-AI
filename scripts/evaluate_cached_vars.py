"""Evaluate a cached-feature VARS checkpoint."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.feature_cache import CachedFeatureDataset, collate_cached_features
from src.models.cached_vars_model import CachedVARSModel
from src.training.trainer import evaluate
from src.utils.runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    device = args.device if args.device != 'auto' else (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    data = CachedFeatureDataset(args.cache)
    loader = DataLoader(data, batch_size=config['training']['eval_batch_size'],
                        collate_fn=collate_cached_features)
    model = CachedVARSModel(**config['model']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    metrics, predictions = evaluate(model, loader, device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'predictions.json', predictions)
    print('Wrote', output)


if __name__ == '__main__':
    main()
