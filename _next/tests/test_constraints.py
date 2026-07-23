import torch

from remapgnn_next.constraints import correction_residuals, project_correction


def test_projection_constraints_and_finite_cg(synthetic_pair):
    pair = synthetic_pair
    torch.manual_seed(1)
    raw = torch.randn(3, pair.fv_operator.n_edges, dtype=torch.float64)
    projected, info = project_correction(
        raw, pair.src_index, pair.tgt_index, pair.area_tgt,
        pair.n_src, pair.n_tgt, iterations=200, assert_converged=True,
        return_info=True,
    )
    row, column = correction_residuals(
        projected, pair.src_index, pair.tgt_index, pair.area_tgt,
        pair.n_src, pair.n_tgt,
    )
    assert info.iterations < 200
    assert row.abs().max() < 1.0e-8
    assert column.abs().max() < 1.0e-10


def test_projection_is_self_adjoint(synthetic_pair):
    pair = synthetic_pair
    x = torch.randn(pair.fv_operator.n_edges, dtype=torch.float64)
    y = torch.randn_like(x)
    px = project_correction(x, pair.src_index, pair.tgt_index, pair.area_tgt,
                            pair.n_src, pair.n_tgt, iterations=200)
    py = project_correction(y, pair.src_index, pair.tgt_index, pair.area_tgt,
                            pair.n_src, pair.n_tgt, iterations=200)
    assert torch.allclose(torch.dot(px, y), torch.dot(x, py), atol=1.0e-10, rtol=1.0e-10)


def test_projection_backward_matches_projection(synthetic_pair):
    pair = synthetic_pair
    raw = torch.randn(pair.fv_operator.n_edges, dtype=torch.float64, requires_grad=True)
    incoming = torch.randn_like(raw)
    projected = project_correction(raw, pair.src_index, pair.tgt_index, pair.area_tgt,
                                   pair.n_src, pair.n_tgt, iterations=200)
    (projected * incoming).sum().backward()
    expected = project_correction(incoming, pair.src_index, pair.tgt_index, pair.area_tgt,
                                  pair.n_src, pair.n_tgt, iterations=200)
    assert torch.allclose(raw.grad, expected, atol=1.0e-10, rtol=1.0e-10)
