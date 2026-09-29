import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.datasets.annotations import FOUL_CLASSES, SEVERITY_CLASSES
from src.datasets.mvfoul import MVFoulDataset, collate_multiview
from src.models.vars_model import VARSModel
from src.training.trainer import evaluate
from src.utils.runtime import seed_everything, write_json


def main():
    p = argparse.ArgumentParser(description='Evaluate a saved configured model; no pretrained download needed')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--split', choices=['Valid', 'Test', 'Chall'], default='Test')
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg = checkpoint['config']
    seed_everything(cfg['seed'])
    model = VARSModel(**{**cfg['model'], 'pretrained': False}).to(args.device)
    model.load_state_dict(checkpoint['model'], strict=True)
    data = MVFoulDataset(args.data_root, args.split, **cfg['data'])
    loader = DataLoader(data, batch_size=cfg['training']['eval_batch_size'],
                        num_workers=cfg['training']['workers'], collate_fn=collate_multiview)
    metrics, predictions = evaluate(model, loader, args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'metrics.json', metrics)
    write_json(output / 'dataset_report.json', data.report)
    write_json(output / 'predictions.json', {'foul_class_order': FOUL_CLASSES,
        'severity_class_order': SEVERITY_CLASSES, 'predictions': predictions})
    print('Wrote', output)


if __name__ == '__main__':
    main()
