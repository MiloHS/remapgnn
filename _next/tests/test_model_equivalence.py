try:
    import pytest
except ImportError:  # unittest discovery imports these pytest-style golden tests
    import unittest
    class _Mark:
        @staticmethod
        def parametrize(*_args, **_kwargs):
            return lambda function: function
    class _Pytest:
        mark = _Mark()
        @staticmethod
        def skip(message):
            raise unittest.SkipTest(message)
    pytest = _Pytest()
import torch

from remapgnn_next.config import StageConfig
from remapgnn_next.progressive import ConservativeCorrectionStage, ProgressiveRemapper


def stage(name="test"):
    config = StageConfig(name=name, band_lower=1.0, band_upper=1.25, edge_dim=8)
    result = ConservativeCorrectionStage(config)
    with torch.no_grad():
        result.score_mlp.net[-1].weight.normal_(std=0.05)
        result.score_mlp.net[-1].bias.normal_(std=0.05)
    return result


def test_forced_rejection_is_exact(synthetic_pair):
    model = ProgressiveRemapper(synthetic_pair.fv_operator, [stage()])
    source = torch.randn(3, synthetic_pair.n_src)
    output, diagnostic = model(synthetic_pair, source, gate_modes=["forced_closed"])
    assert torch.equal(output, diagnostic.fv_output)
    assert torch.count_nonzero(diagnostic.stages[0].delta_weight) == 0


@pytest.mark.parametrize("scale", [-2.5, 1.0e-9])
def test_affine_equivariance_including_negative_and_tiny_scales(synthetic_pair, scale):
    model = ProgressiveRemapper(synthetic_pair.fv_operator, [stage()]).eval()
    source = torch.randn(2, synthetic_pair.n_src)
    offset = 3.25
    first, _ = model(synthetic_pair, source, gate_modes=["forced_open"])
    transformed, _ = model(synthetic_pair, scale * source + offset, gate_modes=["forced_open"])
    expected = scale * first + offset
    assert torch.allclose(transformed, expected, atol=2.0e-6, rtol=2.0e-5)


def test_rotation_invariance(synthetic_pair):
    model = ProgressiveRemapper(synthetic_pair.fv_operator, [stage()]).eval()
    source = torch.randn(2, synthetic_pair.n_src)
    angle = 0.73
    rotation = torch.tensor([
        [torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle)), 0.0],
        [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle)), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rotated = synthetic_pair.__class__(
        **{**synthetic_pair.__dict__,
           "src_xyz": synthetic_pair.src_xyz @ rotation.T,
           "tgt_xyz": synthetic_pair.tgt_xyz @ rotation.T}
    )
    first, _ = model(synthetic_pair, source, gate_modes=["forced_open"])
    second, _ = model(rotated, source, gate_modes=["forced_open"])
    assert torch.allclose(first, second, atol=2.0e-6, rtol=2.0e-6)

