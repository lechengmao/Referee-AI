import numpy as np
import pytest
from src.evaluation.metrics import classification_metrics, multitask_metrics


def test_known_confusion_and_absent_class():
    m = classification_metrics([0, 0, 1, 2], [0, 1, 1, 1], ['a', 'b', 'c', 'd'])
    assert m['confusion_matrix'] == [[1, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
    assert m['accuracy'] == 0.5 and m['balanced_accuracy'] == 0.5
    assert m['per_class']['b']['precision'] == pytest.approx(1/3)
    assert m['per_class']['d']['recall'] is None
    assert m['balanced_accuracy_all_classes'] is None


def test_severity_diagnostics():
    labels = [[0, 3], [1, 3], [2, 2], [3, 0]]
    result = multitask_metrics(labels, np.eye(8)[[0, 1, 2, 3]], np.eye(4)[[3, 2, 2, 0]])
    d = result['screening_diagnostics']
    assert d['red_card_recall'] == d['serious_foul_recall'] == 0.5
    assert d['yellow_card_recall'] == 1 and d['yellow_or_red_detection_recall'] == 1
    assert d['false_alerts_per_hour'] is None
