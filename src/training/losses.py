import torch
from torch.nn import functional as F


def _classification_loss(logits, targets, weights, focal_gamma):
    if not focal_gamma:
        return F.cross_entropy(logits, targets, weight=weights)
    ce = F.cross_entropy(logits, targets, weight=weights, reduction='none')
    probability = logits.softmax(-1).gather(1, targets[:, None]).squeeze(1)
    return (((1 - probability) ** focal_gamma) * ce).mean()


def multitask_loss(output, labels, class_weights=None, settings=None):
    if labels is None or labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError('Training requires B,2 labels')
    weights = class_weights or (None, None)
    settings = settings or {}
    gamma = settings.get('focal_gamma', 0.0)
    foul = _classification_loss(output['foul_logits'], labels[:, 0], weights[0], gamma)
    severity = _classification_loss(output['severity_logits'], labels[:, 1], weights[1], gamma)
    auxiliary = output.get('auxiliary_loss', 0.0)
    return (settings.get('foul_task_weight', 1.0) * foul
            + settings.get('severity_task_weight', 1.0) * severity
            + auxiliary)


def class_frequency_weights(records, device, mode='inverse_frequency'):
    result = []
    for column, count in ((0, 8), (1, 4)):
        labels = torch.tensor([x['labels'][column] for x in records])
        support = torch.bincount(labels, minlength=count).float()
        if (support == 0).any():
            raise ValueError('Weighted reference run requires all classes in training split')
        if mode == 'inverse_frequency':
            weights = support.sum() / support
        elif mode == 'sqrt_inverse_frequency':
            weights = (support.sum() / support).sqrt()
        else:
            raise ValueError(f'Unsupported class weight mode: {mode}')
        # Keep the average loss scale comparable with an unweighted run.
        result.append((weights / weights.mean()).to(device))
    return result


def inverse_frequency_weights(records, device):
    """Backward-compatible weights used by the reference baseline runner."""
    return class_frequency_weights(records, device, 'inverse_frequency')
