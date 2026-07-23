import torch

from remapgnn_next.sparse import apply_operator, index_sum


def test_index_sum_and_operator_application(synthetic_pair):
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    index = torch.tensor([0, 1, 0, 1])
    assert torch.equal(index_sum(values, index, 2), torch.tensor([4.0, 6.0]))
    source = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = apply_operator(synthetic_pair.fv_operator, source)
    assert torch.allclose(result, torch.full((3,), 2.5, dtype=result.dtype))
