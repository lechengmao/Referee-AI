import pytest
import torch
from torch.utils.data import DataLoader

from src.datasets.mvfoul import collate_multiview
from src.models.dt_gcva import DTGCVAFusion
from src.models.vars_model import VARSModel
from src.training.trainer import evaluate
from src.training.losses import multitask_loss
from src.utils.runtime import load_config
from src.utils.synthetic import TinyTestEncoder, SyntheticIncidents


def test_dt_gcva_shapes_masks_weights_and_gradients():
    torch.manual_seed(7)
    layer = DTGCVAFusion(12, fusion_dim=8, num_heads=2, dropout=0, view_dropout=0).double()
    features = torch.randn(2, 3, 12, dtype=torch.double, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    result = layer(features, mask)
    assert result['foul_features'].shape == (2, 12)
    assert result['severity_features'].shape == (2, 12)
    for key in ('foul_attention', 'severity_attention', 'attention'):
        assert result[key].shape == (2, 3)
        torch.testing.assert_close(result[key].sum(1), torch.ones(2, dtype=torch.double))
        assert result[key][0, 2] == 0
    (result['foul_features'].square().sum() + result['severity_features'].square().sum()).backward()
    assert torch.isfinite(features.grad).all() and features.grad[0, 2].abs().sum() == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


def test_dt_gcva_is_permutation_invariant():
    torch.manual_seed(3)
    layer = DTGCVAFusion(8, fusion_dim=8, num_heads=2, dropout=0, view_dropout=0).eval()
    features = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    result = layer(features, mask)
    permutation = torch.tensor([2, 0, 3, 1])
    shuffled = layer(features[:, permutation], mask[:, permutation])
    for key in ('foul_features', 'severity_features'):
        torch.testing.assert_close(result[key], shuffled[key])
    for key in ('foul_attention', 'severity_attention', 'attention'):
        torch.testing.assert_close(result[key][:, permutation], shuffled[key])


def test_view_dropout_always_keeps_a_view():
    torch.manual_seed(5)
    layer = DTGCVAFusion(8, fusion_dim=8, num_heads=2, dropout=0, view_dropout=0.999).train()
    result = layer(torch.randn(4, 3, 8), torch.ones(4, 3, dtype=torch.bool))
    assert torch.isfinite(result['foul_features']).all()
    torch.testing.assert_close(result['attention'].sum(1), torch.ones(4))


def test_dt_gcva_end_to_end_dual_task_gradient():
    torch.manual_seed(11)
    model = VARSModel(encoder=TinyTestEncoder(), hidden_dim=16,
                      aggregation_type='dt_gcva', fusion_dim=8,
                      fusion_heads=2, fusion_dropout=0, view_dropout=0)
    video = torch.randn(2, 3, 3, 4, 8, 8)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    output = model(video, mask)
    assert output['foul_logits'].shape == (2, 8)
    assert output['severity_logits'].shape == (2, 4)
    loss = multitask_loss(output, torch.tensor([[0, 1], [4, 2]]))
    loss.backward()
    assert model.aggregation.task_queries.grad.abs().sum() > 0
    assert model.heads.foul[-1].weight.grad.abs().sum() > 0
    assert model.heads.severity[-1].weight.grad.abs().sum() > 0


def test_dt_gcva_rejects_invalid_configuration_and_empty_views():
    with pytest.raises(ValueError):
        DTGCVAFusion(8, fusion_dim=7, num_heads=2)
    layer = DTGCVAFusion(8, fusion_dim=8, num_heads=2)
    with pytest.raises(ValueError):
        layer(torch.randn(1, 2, 8), torch.zeros(1, 2, dtype=torch.bool))


def test_phase2_config_checkpoint_and_prediction_contract(tmp_path):
    config = load_config('configs/dt_gcva.yaml')
    assert config['model']['aggregation_type'] == 'dt_gcva'
    small = {**config['model'], 'pretrained': False, 'fusion_dim': 8,
             'fusion_heads': 2, 'fusion_dropout': 0, 'view_dropout': 0}
    model = VARSModel(encoder=TinyTestEncoder(), **small)
    checkpoint = tmp_path / 'dt_gcva.pt'
    torch.save({'config': config, 'model': model.state_dict()}, checkpoint)
    state = torch.load(checkpoint, weights_only=True)
    restored = VARSModel(encoder=TinyTestEncoder(), **small)
    restored.load_state_dict(state['model'], strict=True)
    loader = DataLoader(SyntheticIncidents(), batch_size=3, collate_fn=collate_multiview)
    metrics, predictions = evaluate(restored, loader, 'cpu')
    assert metrics is not None and len(predictions) == 8
    assert {'view_weights', 'foul_view_weights', 'severity_view_weights'} <= predictions[0].keys()
    assert sum(predictions[0]['foul_view_weights']) == pytest.approx(1)
    assert sum(predictions[0]['severity_view_weights']) == pytest.approx(1)
