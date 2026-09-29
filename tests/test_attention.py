import pytest
import torch
from src.models.vars_attention import VARSAttention


def test_equations_by_hand():
    layer = VARSAttention(2)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(2))
    f = torch.tensor([[[1., 0.], [1., 1.]]])
    pooled, weights = layer(f)
    # Gram=[[1,1],[1,2]], total=5, row sums=[2,3].
    torch.testing.assert_close(weights, torch.tensor([[0.4, 0.6]]))
    torch.testing.assert_close(pooled, torch.tensor([[1., 0.6]]))


@pytest.mark.parametrize('projected', [False, True])
def test_mask_permutation_and_gradients(projected):
    torch.manual_seed(2)
    layer = VARSAttention(4, projected).double()
    f = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    output, weights = layer(f, mask)
    perm = torch.tensor([2, 0, 1])
    shuffled, w = layer(f[:, perm], mask[:, perm])
    torch.testing.assert_close(output, shuffled)
    torch.testing.assert_close(w, weights[:, perm])
    torch.testing.assert_close(weights.sum(1), torch.ones(2, dtype=torch.double))
    assert weights[0, 2] == 0
    output.square().sum().backward()
    assert torch.isfinite(f.grad).all() and torch.isfinite(layer.weight.grad).all()
    assert f.grad[0, 2].abs().sum() == 0


def test_zero_and_single_view():
    layer = VARSAttention(2)
    output, weights = layer(torch.zeros(1, 3, 2), torch.tensor([[True, True, False]]))
    torch.testing.assert_close(weights, torch.tensor([[0.5, 0.5, 0.]]))
    assert torch.isfinite(output).all()
    f = torch.randn(1, 1, 2)
    pooled, w = layer(f)
    torch.testing.assert_close(pooled, f[:, 0])
    assert w.shape == (1, 1) and w.item() == 1


def test_no_views_rejected():
    with pytest.raises(ValueError):
        VARSAttention(2)(torch.zeros(1, 2, 2), torch.zeros(1, 2, dtype=torch.bool))


def test_original_vs_projected_values():
    f = torch.tensor([[[1., 2.], [2., 3.]]])
    paper, reference = VARSAttention(2), VARSAttention(2, True)
    with torch.no_grad():
        paper.weight.copy_(torch.eye(2) * 2)
        reference.weight.copy_(paper.weight)
    a, aw = paper(f)
    b, bw = reference(f)
    torch.testing.assert_close(aw, bw)
    torch.testing.assert_close(b, a * 2)
