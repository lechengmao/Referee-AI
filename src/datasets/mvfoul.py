"""Video decoding, temporally consistent transforms, variable-view batching."""
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.video import MViT_V2_S_Weights
from .annotations import read_split


def sample_indices(frame_count, fps, center_seconds, frames=16, duration=1.0, shift=0.0):
    if fps <= 0 or frame_count <= 0 or frames < 1 or duration <= 0:
        raise ValueError('Invalid video metadata or temporal sampling settings')
    # Half-open window: 8 samples before the center, 8 at/after it.
    times = center_seconds + shift + (np.arange(frames) / frames - 0.5) * duration
    indices = np.rint(times * fps).astype(int)
    if indices.min() < 0 or indices.max() >= frame_count:
        raise ValueError('Sampling window is outside video; inspect center_seconds and clip length')
    return indices


def read_frames(path, center_seconds, frames=16, duration=1.0, shift=0.0):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {path}')
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ids = sample_indices(count, fps, center_seconds, frames, duration, shift)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ids[0]))
        selected, wanted = {}, set(ids.tolist())
        for i in range(int(ids[0]), int(ids[-1]) + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Decode failed: {path}, frame {i}')
            if i in wanted:
                selected[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(np.stack([selected[i] for i in ids])).permute(0, 3, 1, 2)
    finally:
        cap.release()


def temporal_window_indices(frame_count, fps, frames=16, window_seconds=1.0,
                            max_windows=5):
    """Return indices for consecutive full, non-overlapping windows."""
    if fps <= 0 or frame_count <= 0 or frames < 1 or window_seconds <= 0 or max_windows < 1:
        raise ValueError('Invalid video metadata or temporal-window settings')
    available = frame_count / fps
    windows = min(max_windows, int(available // window_seconds))
    if windows < 1:
        raise ValueError('Video must contain at least one complete temporal window')
    result = []
    for index in range(windows):
        center = (index + 0.5) * window_seconds
        result.append(sample_indices(frame_count, fps, center, frames, window_seconds))
    return np.stack(result)


def read_temporal_windows(path, frames=16, window_seconds=1.0, max_windows=5):
    """Decode a video once and return W,T,C,H,W uint8 windows."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {path}')
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = temporal_window_indices(count, fps, frames, window_seconds, max_windows)
        wanted = set(indices.reshape(-1).tolist())
        selected = {}
        for frame_index in range(int(indices.max()) + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Decode failed: {path}, frame {frame_index}')
            if frame_index in wanted:
                selected[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        windows = [torch.from_numpy(np.stack([selected[i] for i in row])).permute(0, 3, 1, 2)
                   for row in indices]
        return torch.stack(windows), indices
    finally:
        cap.release()


class MVFoulDataset(Dataset):
    def __init__(self, root, split, center_seconds=3.0, frames=16, duration=1.0,
                 temporal_jitter=0.125, train_views=2, augment=True, subset_ids=None,
                 deterministic=False, deterministic_views=None):
        self.records, self.report = read_split(root, split, subset_ids=subset_ids)
        self.training = split == 'Train' and not deterministic
        self.center, self.frames, self.duration = center_seconds, frames, duration
        self.jitter, self.train_views = temporal_jitter, train_views
        self.deterministic_views = deterministic_views
        if train_views < 1 or temporal_jitter < 0:
            raise ValueError('train_views must be positive and jitter nonnegative')
        if deterministic_views is not None and deterministic_views < 1:
            raise ValueError('deterministic_views must be positive or None')
        self.preprocess = MViT_V2_S_Weights.KINETICS400_V1.transforms()
        # Applied to a T,C,H,W tensor: parameters shared across time per view.
        self.augment = transforms.Compose([
            transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1)),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.5, saturation=0.5, contrast=0.5),
            transforms.RandomHorizontalFlip(),
        ]) if self.training and augment else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        paths = record['paths']
        if self.training:
            paths = random.sample(paths, min(self.train_views, len(paths)))
        elif self.deterministic_views is not None:
            paths = paths[:min(self.deterministic_views, len(paths))]
        # Same temporal shift across manually aligned views.
        shift = random.uniform(-self.jitter, self.jitter) if self.training else 0.0
        clips = []
        for path in paths:
            clip = read_frames(path, self.center, self.frames, self.duration, shift)
            if self.augment is not None:
                clip = self.augment(clip)
            clips.append(self.preprocess(clip))  # C,T,224,224
        return {'video': torch.stack(clips), 'labels': record['labels'],
                'id': record['id'], 'paths': [str(p) for p in paths]}


def collate_multiview(items):
    if not items:
        raise ValueError('Empty batch')
    max_views = max(x['video'].shape[0] for x in items)
    shape = items[0]['video'].shape[1:]
    video = items[0]['video'].new_zeros((len(items), max_views, *shape))
    mask = torch.zeros(len(items), max_views, dtype=torch.bool)
    for i, item in enumerate(items):
        n = item['video'].shape[0]
        if n < 1 or item['video'].shape[1:] != shape:
            raise ValueError('Invalid or inconsistent view shape')
        video[i, :n], mask[i, :n] = item['video'], True
    labelled = [x['labels'] is not None for x in items]
    if any(labelled) and not all(labelled):
        raise ValueError('Cannot mix labelled and unlabelled incidents')
    return {'video': video, 'mask': mask,
            'labels': torch.tensor([x['labels'] for x in items], dtype=torch.long) if all(labelled) else None,
            'ids': [x['id'] for x in items], 'paths': [x['paths'] for x in items]}
