"""Local HTTP API for real MViT + DT-GCVA inference."""
import os
import tempfile
import threading
from pathlib import Path

import cv2
import torch
from flask import Flask, jsonify, request
from torchvision.models.video import MViT_V2_S_Weights

from src.datasets.mvfoul import read_temporal_windows
from src.models.mvit_encoder import MViTEncoder
from src.models.spatiotemporal_model import SpatiotemporalDTGCVAModel


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'experiments/spatiotemporal-balanced-full-20260928/best.pt'
FOUL_CLASSES = ['Tackling', 'Standing tackling', 'High leg', 'Holding',
                'Pushing', 'Elbowing', 'Challenge', 'Dive/Simulation']
SEVERITY_CLASSES = ['No offence', 'Offence + No card',
                    'Offence + Yellow card', 'Offence + Red card']
ALLOWED_SUFFIXES = {'.mp4', '.mov', '.webm'}
MAX_FILE_BYTES = 80 * 1024 * 1024
MAX_DURATION_SECONDS = 5.15

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * MAX_FILE_BYTES + 1024 * 1024
_runtime = None
_load_lock = threading.Lock()
_inference_lock = threading.Lock()


def _device():
    return ('cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else 'cpu')


def _load_runtime():
    global _runtime
    if _runtime is not None:
        return _runtime
    with _load_lock:
        if _runtime is None:
            device = _device()
            checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
            model = SpatiotemporalDTGCVAModel(
                **checkpoint['config']['model']).eval().to(device)
            model.load_state_dict(checkpoint['model'], strict=True)
            encoder = MViTEncoder(
                pretrained=True, representation='embedding').eval().to(device)
            preprocess = MViT_V2_S_Weights.KINETICS400_V1.transforms()
            _runtime = (device, encoder, model, preprocess)
    return _runtime


def _duration(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError('One of the uploaded videos could not be decoded.')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or frames <= 0:
            raise ValueError('One of the uploaded videos has invalid metadata.')
        return frames / fps
    finally:
        cap.release()


def _top(probabilities, labels):
    order = sorted(range(len(labels)), key=probabilities.__getitem__, reverse=True)[:3]
    return [{'label': labels[index], 'value': probabilities[index]} for index in order]


def _confidence(foul, severity):
    top = min(max(foul), max(severity))
    return 'high' if top >= .70 else 'moderate' if top >= .50 else 'low'


@torch.inference_mode()
def infer(paths):
    device, encoder, model, preprocess = _load_runtime()
    raw_views = [read_temporal_windows(path)[0] for path in paths]
    common_windows = min(len(windows) for windows in raw_views)
    if common_windows < 1:
        raise ValueError('Each video must contain at least one complete second.')
    encoded_views = []
    for raw in raw_views:
        windows = torch.stack([preprocess(window) for window in raw[:common_windows]])
        chunks = [encoder(chunk.to(device)).cpu() for chunk in windows.split(2)]
        encoded_views.append(torch.cat(chunks))
    features = torch.stack(encoded_views).unsqueeze(0).float().to(device)
    mask = torch.ones(features.shape[:3], dtype=torch.bool, device=device)
    output = model(features, mask)
    foul = output['foul_logits'][0].softmax(-1).cpu().tolist()
    severity = output['severity_logits'][0].softmax(-1).cpu().tolist()
    foul_views = output['foul_view_attention'][0].cpu()
    severity_views = output['severity_view_attention'][0].cpu()
    views = ((foul_views + severity_views) / 2).tolist()
    total = sum(views)
    views = [value / total for value in views]
    return {
        'id': 'live-upload',
        'title': 'Uploaded incident',
        'subtitle': f'{len(paths)} uploaded view' + ('s' if len(paths) != 1 else '') +
                    f' · {common_windows} temporal windows · live inference',
        'foul': _top(foul, FOUL_CLASSES),
        'severity': _top(severity, SEVERITY_CLASSES),
        'temporalWeights': [],
        'viewWeights': views,
        'confidence': _confidence(foul, severity),
        'note': 'Generated locally by the frozen MViT + DT-GCVA checkpoint. Interpret as research evidence, not an officiating decision.',
    }


@app.get('/api/health')
def health():
    return jsonify({'status': 'ok', 'model': 'DT-GCVA', 'device': _device()})


@app.post('/api/infer')
def api_infer():
    files = request.files.getlist('videos')
    if not 1 <= len(files) <= 4:
        return jsonify({'error': 'Upload between 1 and 4 video views.'}), 400
    try:
        with tempfile.TemporaryDirectory(prefix='referee-ai-') as directory:
            paths = []
            for index, uploaded in enumerate(files):
                suffix = Path(uploaded.filename or '').suffix.lower()
                if suffix not in ALLOWED_SUFFIXES:
                    raise ValueError('Only MP4, MOV and WebM videos are supported.')
                path = Path(directory) / f'view-{index}{suffix}'
                uploaded.save(path)
                if path.stat().st_size > MAX_FILE_BYTES:
                    raise ValueError('Each video must be 80 MB or smaller.')
                if _duration(path) > MAX_DURATION_SECONDS:
                    raise ValueError('Each video must be 5 seconds or shorter.')
                paths.append(path)
            if not _inference_lock.acquire(blocking=False):
                return jsonify({'error': 'The model is already analysing another incident. Please retry shortly.'}), 429
            try:
                return jsonify(infer(paths))
            finally:
                _inference_lock.release()
    except ValueError as error:
        return jsonify({'error': str(error)}), 400
    except Exception:
        app.logger.exception('Inference failed')
        return jsonify({'error': 'Model inference failed. Check the API log and try again.'}), 500


if __name__ == '__main__':
    os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
    app.run(host='127.0.0.1', port=8000, threaded=True)
