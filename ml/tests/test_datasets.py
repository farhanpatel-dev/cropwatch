import numpy as np

from cropwatch_ml.datasets import IGNORE_INDEX, build_target, field_id, split_by_field

CLASS_IDS = {"background": 0, "nutrient_deficiency": 1, "drydown": 2, "weed_cluster": 3}


def _mask(*pixels: tuple[int, int]) -> np.ndarray:
    m = np.zeros((2, 3), dtype=bool)
    for r, c in pixels:
        m[r, c] = True
    return m


def test_build_target_priority_ignore_and_unknown_classes() -> None:
    labels = {
        "drydown": _mask((0, 0), (0, 1), (0, 2)),
        "nutrient_deficiency": _mask((0, 1), (0, 2)),
        "weed_cluster": _mask((0, 2)),
        "endrow": _mask((1, 0)),  # not one of our classes -> stays background
    }
    valid = np.ones((2, 3), dtype=bool)
    valid[1, 2] = False
    target = build_target(labels, valid, CLASS_IDS)
    assert target.tolist() == [[2, 1, 3], [0, 0, IGNORE_INDEX]]


def test_ignore_wins_over_labels() -> None:
    valid = np.zeros((2, 3), dtype=bool)
    target = build_target({"weed_cluster": np.ones((2, 3), dtype=bool)}, valid, CLASS_IDS)
    assert (target == IGNORE_INDEX).all()


def test_split_by_field_keeps_fields_together() -> None:
    keys = [f"F{f}_{i}-0-{i + 512}-512" for f in range(10) for i in range(3)]
    train, val = split_by_field(keys, n_val_fields=3, seed=0)
    assert len(train) + len(val) == len(keys)
    assert {field_id(k) for k in train}.isdisjoint({field_id(k) for k in val})
    assert len({field_id(k) for k in val}) == 3
    assert split_by_field(keys, n_val_fields=3, seed=0) == (train, val)  # same seed, same split
