"""Fixed-size confusion matrices; absent-class recall is null, not fabricated."""
import numpy as np
from src.datasets.annotations import FOUL_CLASSES, SEVERITY_CLASSES


def classification_metrics(target, prediction, classes):
    target, prediction = np.asarray(target, dtype=int), np.asarray(prediction, dtype=int)
    n = len(classes)
    if target.ndim != 1 or target.shape != prediction.shape or not len(target):
        raise ValueError('Expected nonempty, equally sized target/prediction vectors')
    if (target < 0).any() or (target >= n).any() or (prediction < 0).any() or (prediction >= n).any():
        raise ValueError('Class ID outside fixed label space')
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (target, prediction), 1)
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(n, dtype=float), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(n, dtype=float), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(n, dtype=float), where=precision + recall > 0)
    present = support > 0
    return {'accuracy': float(tp.sum() / cm.sum()),
            'balanced_accuracy': float(recall[present].mean()),
            'balanced_accuracy_all_classes': float(recall.mean()) if present.all() else None,
            'macro_precision': float(precision.mean()), 'macro_recall_zero_filled': float(recall.mean()),
            'macro_f1': float(f1.mean()), 'weighted_f1': float(np.dot(f1, support) / support.sum()),
            'confusion_matrix': cm.tolist(), 'class_order': list(classes),
            'absent_classes': [classes[i] for i in range(n) if not present[i]],
            'per_class': {name: {'precision': float(precision[i]),
                                 'recall': float(recall[i]) if support[i] else None,
                                 'f1': float(f1[i]), 'support': int(support[i]),
                                 'predicted': int(predicted[i])}
                          for i, name in enumerate(classes)}}


def multitask_metrics(labels, foul_probs, severity_probs):
    labels, fp, sp = np.asarray(labels), np.asarray(foul_probs), np.asarray(severity_probs)
    if labels.shape != (len(fp), 2) or fp.shape != (len(labels), 8) or sp.shape != (len(labels), 4):
        raise ValueError('Expected N,2 labels and N,8 / N,4 probabilities')
    for p in (fp, sp):
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
            raise ValueError('Invalid probabilities')
    pred = sp.argmax(1)
    result = {'foul': classification_metrics(labels[:, 0], fp.argmax(1), FOUL_CLASSES),
              'severity': classification_metrics(labels[:, 1], pred, SEVERITY_CLASSES)}
    per = result['severity']['per_class']
    positive = labels[:, 1] == 3
    # Phase 1 convention: "serious" means the red-card target only.
    # Also report yellow-or-red separately, without equating it to VAR eligibility.
    card = labels[:, 1] >= 2
    result['screening_diagnostics'] = {
        'definition': 'serious = red-card label; argmax classification, no alert threshold/calibration',
        'yellow_card_recall': per[SEVERITY_CLASSES[2]]['recall'],
        'red_card_recall': per[SEVERITY_CLASSES[3]]['recall'],
        'serious_foul_recall': float((pred[positive] == 3).mean()) if positive.any() else None,
        'yellow_or_red_detection_recall': float((pred[card] >= 2).mean()) if card.any() else None,
        'false_alerts_per_hour': None,
        'deployment_note': 'Cannot estimate match-level alert rates from pre-cut incident clips.'}
    return result
