"""Cache deterministic five-window MViT embeddings for one official split."""
import argparse
import hashlib
from pathlib import Path

import torch
from torchvision.models.video import MViT_V2_S_Weights

from scripts.overfit_subset import EXPECTED
from src.datasets.annotations import read_split
from src.datasets.mvfoul import read_temporal_windows
from src.models.mvit_encoder import MViTEncoder
from src.utils.runtime import environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data/SoccerNet/mvfouls')
    parser.add_argument('--split', choices=['Train', 'Valid', 'Test'], default='Train')
    parser.add_argument('--output', required=True)
    parser.add_argument('--real53', action='store_true')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--encoder-batch-size', type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output)
    partial = output.with_suffix(output.suffix + '.partial')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    if args.real53 and args.split != 'Train':
        raise ValueError('--real53 is only valid for Train')
    if args.encoder_batch_size < 1:
        raise ValueError('--encoder-batch-size must be positive')
    device = args.device if args.device != 'auto' else (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu')
    root = Path(args.data_root).resolve()
    records, report = read_split(root, args.split,
        subset_ids=EXPECTED if args.real53 else None)
    annotation = root / args.split / 'annotations.json'
    metadata = {'split': args.split, 'real53': args.real53,
                'window_seconds': 1.0, 'max_windows': 5, 'frames_per_window': 16,
                'view_window_alignment': 'longest_common_full-window-prefix',
                'encoder': 'torchvision_mvit_v2_s_KINETICS400_V1_embedding',
                'annotation_sha256': hashlib.sha256(annotation.read_bytes()).hexdigest(),
                'dataset_report': report, 'environment': environment()}
    cached = []
    if partial.exists():
        resume = torch.load(partial, map_location='cpu', weights_only=True)
        if (resume.get('format_version') != 1 or
                resume.get('metadata', {}).get('annotation_sha256') != metadata['annotation_sha256']):
            raise ValueError(f'Incompatible partial cache: {partial}')
        cached = resume['records']
        expected = [record['id'] for record in records[:len(cached)]]
        if [record['id'] for record in cached] != expected:
            raise ValueError('Partial cache is not a prefix of the requested split')
        print(f'resuming {len(cached)}/{len(records)} from {partial}', flush=True)
    encoder = MViTEncoder(pretrained=True, representation='embedding').eval().to(device)
    preprocess = MViT_V2_S_Weights.KINETICS400_V1.transforms()
    with torch.inference_mode():
        for record_index, record in enumerate(records[len(cached):], start=len(cached)):
            raw_views = []
            for path in record['paths']:
                raw, _ = read_temporal_windows(path)
                raw_views.append(raw)
            # Official camera clips can differ by a fraction of a second. Keep
            # only the common aligned prefix; never pad a missing window or use
            # future evidence from one camera alone.
            common_windows = min(len(raw) for raw in raw_views)
            if common_windows < 1:
                raise ValueError(f'No common full window in {record["id"]}')
            starts = [float(i) for i in range(common_windows)]
            view_tensors = []
            for raw in raw_views:
                windows = torch.stack([
                    preprocess(window) for window in raw[:common_windows]])
                encoded = []
                for chunk in windows.split(args.encoder_batch_size):
                    encoded.append(encoder(chunk.to(device)).cpu())
                view_tensors.append(torch.cat(encoded))
            cached.append({'id': record['id'], 'paths': [str(p) for p in record['paths']],
                           'labels': record['labels'], 'window_starts': starts,
                           'features': torch.stack(view_tensors)})
            if (record_index + 1) % 10 == 0 or record_index + 1 == len(records):
                print(f'cached {record_index + 1}/{len(records)} {record["id"]}', flush=True)
            if (record_index + 1) % 25 == 0:
                torch.save({'format_version': 1, 'metadata': metadata, 'records': cached}, partial)
    payload = {'format_version': 1, 'metadata': metadata, 'records': cached}
    torch.save(payload, output)
    if partial.exists():
        partial.unlink()
    print('wrote', output, flush=True)


if __name__ == '__main__':
    main()
