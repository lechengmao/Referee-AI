"""Evaluate a saved checkpoint on the fixed 53-incident Train subset."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts.overfit_subset import EXPECTED
from src.datasets.mvfoul import MVFoulDataset, collate_multiview
from src.models.vars_model import VARSModel
from src.training.trainer import evaluate
from src.utils.runtime import seed_everything, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/SoccerNet/mvfouls')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    device = args.device if args.device != 'auto' else (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    seed_everything(config['seed'])
    data = MVFoulDataset(args.data_root, 'Train', subset_ids=EXPECTED, **config['data'])
    loader = DataLoader(data, batch_size=config['training']['eval_batch_size'],
                        num_workers=config['training']['workers'], collate_fn=collate_multiview)
    model = VARSModel(**{**config['model'], 'pretrained': False}).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    metrics, predictions = evaluate(model, loader, device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'predictions.json', predictions)
    write_json(output / 'run.json', {'checkpoint': str(Path(args.checkpoint).resolve()),
        'device': device, 'incidents': len(data), 'train_subset_evaluation': True})
    print(f"TRAIN53 loss={metrics['loss_unweighted']:.6f} "
          f"foul={metrics['foul']['accuracy']:.4f} "
          f"severity={metrics['severity']['accuracy']:.4f}", flush=True)


if __name__ == '__main__':
    main()
