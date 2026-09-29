import torch
from torch.utils.data import DataLoader

from src.datasets.feature_cache import CachedFeatureDataset, collate_cached_features
from src.models.spatiotemporal_model import SpatiotemporalDTGCVAModel
from src.models.cached_vars_model import CachedVARSModel
from src.training.losses import multitask_loss
from src.training.trainer import evaluate, train_epoch


def make_cache(path):
    records = []
    for index, (views, windows) in enumerate(((1, 3), (2, 5), (3, 4))):
        records.append({'id': f'Train/{index}', 'paths': [f'v{v}' for v in range(views)],
                        'labels': (index, index % 4),
                        'window_starts': [float(t) for t in range(windows)],
                        'features': torch.randn(views, windows, 12)})
    torch.save({'format_version': 1, 'metadata': {'split': 'Train'}, 'records': records}, path)


def test_cached_collation_and_spatiotemporal_outputs(tmp_path):
    path = tmp_path / 'cache.pt'
    make_cache(path)
    dataset = CachedFeatureDataset(path)
    batch = collate_cached_features([dataset[0], dataset[1]])
    assert batch['features'].shape == (2, 2, 5, 12)
    assert batch['mask'].sum(1).sum(1).tolist() == [3, 10]
    model = SpatiotemporalDTGCVAModel(input_dim=12, hidden_dim=16,
        fusion_dim=8, fusion_heads=2, fusion_dropout=0, token_dropout=0)
    output = model(batch['features'], batch['mask'])
    assert output['foul_attention'].shape == (2, 2, 5)
    torch.testing.assert_close(output['foul_attention'].sum((1, 2)), torch.ones(2))
    torch.testing.assert_close(output['severity_attention'].sum((1, 2)), torch.ones(2))
    assert output['foul_attention'][0, 1].sum() == 0
    assert output['auxiliary_loss'].ndim == 0 and output['auxiliary_loss'] >= 0
    assert output['foul_attention_entropy'].shape == (2,)
    assert output['severity_attention_entropy'].shape == (2,)
    multitask_loss(output, batch['labels']).backward()
    assert model.aggregation.task_queries.grad.abs().sum() > 0


def test_cached_training_evaluation_and_prediction_contract(tmp_path):
    path = tmp_path / 'cache.pt'
    make_cache(path)
    loader = DataLoader(CachedFeatureDataset(path), batch_size=2,
                        collate_fn=collate_cached_features)
    model = SpatiotemporalDTGCVAModel(input_dim=12, hidden_dim=16,
        fusion_dim=8, fusion_heads=2, fusion_dropout=0, token_dropout=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    assert train_epoch(model, loader, optimizer, 'cpu') > 0
    metrics, predictions = evaluate(model, loader, 'cpu')
    assert metrics is not None and len(predictions) == 3
    assert len(predictions[0]['foul_spatiotemporal_weights']) == 1
    assert len(predictions[0]['foul_spatiotemporal_weights'][0]) == 3
    assert len(predictions[1]['severity_temporal_weights']) == 5


def test_spatiotemporal_view_and_time_permutation_invariance():
    torch.manual_seed(19)
    model = SpatiotemporalDTGCVAModel(input_dim=12, hidden_dim=16,
        fusion_dim=8, fusion_heads=2, fusion_dropout=0, token_dropout=0).eval()
    features = torch.randn(1, 2, 5, 12)
    mask = torch.ones(1, 2, 5, dtype=torch.bool)
    original = model(features, mask)
    view_perm, time_perm = torch.tensor([1, 0]), torch.tensor([3, 0, 4, 1, 2])
    permuted = model(features[:, view_perm][:, :, time_perm], mask[:, view_perm][:, :, time_perm])
    torch.testing.assert_close(original['foul_logits'], permuted['foul_logits'])
    torch.testing.assert_close(original['severity_logits'], permuted['severity_logits'])


def test_hierarchical_spatiotemporal_attention_contract():
    torch.manual_seed(23)
    model = SpatiotemporalDTGCVAModel(
        input_dim=12, hidden_dim=16, fusion_dim=8, fusion_heads=2,
        fusion_dropout=0, token_dropout=0,
        fusion_strategy='hierarchical', view_temperature=0.25,
        temporal_temperature=0.2).eval()
    features = torch.randn(2, 3, 5, 12)
    mask = torch.ones(2, 3, 5, dtype=torch.bool)
    mask[0, 2] = False
    output = model(features, mask)
    assert output['foul_attention'].shape == (2, 3, 5)
    torch.testing.assert_close(output['foul_attention'].sum((1, 2)), torch.ones(2))
    torch.testing.assert_close(output['severity_attention'].sum((1, 2)), torch.ones(2))
    assert output['foul_attention'][0, 2].sum() == 0
    output['foul_logits'].sum().backward()
    assert model.aggregation.view_queries.grad.abs().sum() > 0
    assert model.aggregation.temporal_queries.grad.abs().sum() > 0


def test_cached_vars_temporal_mean_contract():
    model = CachedVARSModel(input_dim=12, hidden_dim=16)
    features = torch.randn(2, 3, 5, 12)
    mask = torch.ones(2, 3, 5, dtype=torch.bool)
    mask[0, 2] = False
    output = model(features, mask)
    assert output['foul_logits'].shape == (2, 8)
    assert output['severity_logits'].shape == (2, 4)
    assert output['attention'].shape == (2, 3, 5)
    torch.testing.assert_close(output['attention'].sum((1, 2)), torch.ones(2))
    assert output['attention'][0, 2].sum() == 0
