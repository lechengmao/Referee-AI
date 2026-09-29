"""CPU-only synthetic optimizer, validation, checkpoint and reload exercise."""
import argparse
import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.datasets.mvfoul import collate_multiview
from src.models.vars_model import VARSModel
from src.training.trainer import fit, evaluate
from src.utils.runtime import load_config, seed_everything, write_json
from src.utils.synthetic import TinyTestEncoder, SyntheticIncidents


def run(output):
    torch.set_num_threads(2)
    seed_everything(42)
    cfg = load_config('configs/baseline.yaml')
    cfg['name'] = 'SYNTHETIC_ONLY_NOT_RESEARCH_RESULTS'
    cfg['training']['epochs'] = 2
    cfg['model']['pretrained'] = False
    cfg['model']['test_encoder'] = 'TinyTestEncoder'
    loaders = [DataLoader(SyntheticIncidents(seed=s), batch_size=3, collate_fn=collate_multiview)
               for s in (123, 999)]
    model = VARSModel(encoder=TinyTestEncoder(), hidden_dim=16)
    before = model.encoder.net[0].weight.detach().clone()
    best = fit(model, *loaders, cfg, output, 'cpu')
    assert not torch.equal(before, model.encoder.net[0].weight), 'Encoder weights did not update'
    state = torch.load(best, map_location='cpu', weights_only=True)
    reloaded = VARSModel(encoder=TinyTestEncoder(), hidden_dim=16)
    reloaded.load_state_dict(state['model'])
    metrics, predictions = evaluate(reloaded, loaders[1], 'cpu')
    assert len(predictions) == 8
    assert len(metrics['severity']['confusion_matrix']) == 4
    write_json(Path(output) / 'smoke_status.json', {
        'synthetic_only': True, 'encoder_weights_updated': True,
        'optimizer_steps': 6, 'epochs': 2, 'checkpoint_reloaded': True,
        'evaluated_incidents': len(predictions), 'research_result': False})
    print('PASS: synthetic training, validation, checkpoint save/load and probability output')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', help='New run directory; omitted uses a disposable temporary directory')
    args = p.parse_args()
    if args.output:
        run(args.output)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            run(tmp)
