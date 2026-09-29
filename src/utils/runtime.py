import json
import os
import platform
import random
from pathlib import Path
import numpy as np
import torch
import torchvision
import yaml


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def load_config(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    if 'extends' in config:
        base = load_config(path.parent / config.pop('extends'))
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key].update(value)
            else:
                base[key] = value
        config = base
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def environment():
    return {'python': platform.python_version(), 'torch': str(torch.__version__),
            'torchvision': str(torchvision.__version__), 'numpy': np.__version__,
            'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda}
