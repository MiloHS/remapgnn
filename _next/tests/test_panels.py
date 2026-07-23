import torch

from remapgnn_next.fields import source_keyed_mode_split
from remapgnn_next.panels import assert_split_disjoint, band_degrees, safety_degree
from remapgnn_next.types import FieldBatch


def test_source_keyed_splits_are_disjoint_and_target_independent():
    train = set(source_keyed_mode_split("CS-r32", 20, 42, "train"))
    validation = set(source_keyed_mode_split("CS-r32", 20, 42, "validation"))
    audit = set(source_keyed_mode_split("CS-r32", 20, 42, "audit"))
    assert not (train & validation or train & audit or validation & audit)
    assert train | validation | audit == set(range(-20, 21))


def test_panel_source_key_guard():
    def panel(key):
        return FieldBatch(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1),
                          [(1, 0)], ["target"], [key])
    assert_split_disjoint(panel("a"), panel("b"), panel("c"))


def test_lower_boundary_safety_degree_stays_outside_icod_target_band():
    source_k = (10242 / 6.0) ** 0.5
    targets = band_degrees(source_k, 1.25, 1.5)
    guard = safety_degree(source_k, 1.25, 1.25, 1.5)
    assert targets[0] == 52
    assert guard == 51
    assert guard not in targets
    assert guard / source_k <= 1.25
