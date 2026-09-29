import argparse
import hashlib
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.datasets.mvfoul import MVFoulDataset, collate_multiview
from src.models.vars_model import VARSModel
from src.training.trainer import fit
from src.training.losses import inverse_frequency_weights
from src.utils.runtime import load_config, seed_everything, seed_worker, write_json


def main():
    p = argparse.ArgumentParser(description='Train a configured multi-view model on official Train/Valid splits')
    p.add_argument('--config', default='configs/baseline.yaml')
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg['seed'])
    train = MVFoulDataset(args.data_root, 'Train', **cfg['data'])
    valid = MVFoulDataset(args.data_root, 'Valid', **cfg['data'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'dataset_report.json', {'train': train.report, 'valid': valid.report,
        'annotation_sha256': {s: hashlib.sha256((Path(args.data_root)/s/'annotations.json').read_bytes()).hexdigest()
                              for s in ('Train', 'Valid')}})
    options = {'num_workers': cfg['training']['workers'], 'collate_fn': collate_multiview,
               'worker_init_fn': seed_worker, 'pin_memory': args.device.startswith('cuda')}
    train_loader = DataLoader(train, batch_size=cfg['training']['batch_size'], shuffle=True,
        generator=torch.Generator().manual_seed(cfg['seed']), **options)
    valid_loader = DataLoader(valid, batch_size=cfg['training']['eval_batch_size'], **options)
    weights = inverse_frequency_weights(train.records, args.device) if cfg['training']['class_weighted'] else None
    print('Best checkpoint:', fit(VARSModel(**cfg['model']), train_loader, valid_loader,
                                  cfg, output, args.device, weights))


if __name__ == '__main__':
    main()
