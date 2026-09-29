import pytest
import torch
from src.datasets.mvfoul import collate_multiview
from src.models.vars_model import VARSModel
from src.models.mvit_encoder import MViTEncoder
from src.training.losses import multitask_loss
from src.utils.synthetic import TinyTestEncoder, SyntheticIncidents
from scripts.smoke_test import run
from scripts.overfit_subset import criterion_met


def test_full_task_gradient():
    batch = collate_multiview([SyntheticIncidents()[i] for i in (0, 2)])
    model = VARSModel(encoder=TinyTestEncoder(), hidden_dim=16)
    output = model(batch['video'], batch['mask'])
    assert output['foul_logits'].shape == (2, 8)
    assert output['severity_logits'].shape == (2, 4)
    loss = multitask_loss(output, batch['labels'])
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model.encoder.net[0].weight.grad.abs().sum() > 0


def test_train_save_reload(tmp_path):
    run(tmp_path)
    assert (tmp_path / 'best.pt').is_file()
    assert (tmp_path / 'last.pt').is_file()
    assert len((tmp_path / 'history.jsonl').read_text().splitlines()) == 2


def test_separate_overfit_thresholds_and_legacy_criterion():
    metrics={'foul':{'accuracy':0.76}, 'severity':{'accuracy':0.91},
             'loss_unweighted':0.29}
    assert criterion_met(metrics, 1.0, {'foul_accuracy':0.75,
                         'severity_accuracy':0.90, 'loss_reduction':0.70})
    assert not criterion_met(metrics, 1.0, {'accuracy':0.90,
                             'loss_reduction':0.70})


def test_runtime_config_defers_evaluation_until_fresh_process():
    from src.utils.runtime import load_config
    config = load_config('configs/dt_gcva_real53_runtime.yaml')
    assert config['model']['aggregation_type'] == 'dt_gcva'
    assert config['training']['epochs'] == 1
    assert config['training']['skip_post_epoch_evaluation'] is True


@pytest.mark.mvit
@pytest.mark.parametrize('representation,dim', [('embedding', 768), ('kinetics_logits', 400)])
def test_real_mvit_forward(representation, dim):
    # The real torchvision MViTv2-S, random weights, no network download.
    encoder = MViTEncoder(pretrained=False, representation=representation).eval()
    model = VARSModel(encoder=encoder, hidden_dim=32).eval()
    with torch.inference_mode():
        result = model(torch.randn(1, 1, 3, 16, 224, 224), torch.ones(1, 1, dtype=torch.bool))
    assert encoder.output_dim == dim
    assert result['foul_logits'].shape == (1, 8)
    assert result['severity_logits'].shape == (1, 4)
    assert torch.isfinite(result['severity_logits']).all()


@pytest.mark.mvit
def test_real_mvit_backward():
    model = VARSModel(pretrained=False, hidden_dim=32).train()
    video = torch.randn(1, 1, 3, 16, 224, 224)
    output = model(video, torch.ones(1, 1, dtype=torch.bool))
    loss = multitask_loss(output, torch.tensor([[0, 3]]))
    loss.backward()
    gradient = model.encoder.network.conv_proj.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    assert model.heads.severity[-1].weight.grad.abs().sum() > 0
