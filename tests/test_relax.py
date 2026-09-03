"""Tests for `relax` behaviors that are awkward as doctests."""

from aces.config import TaskExtractorConfig

from aces_relaxations import relax


def test_relax_does_not_mutate_input(readmission_cfg: TaskExtractorConfig):
    """`relax` must copy the configuration before applying in-place relaxations."""
    original_has = {n: dict(w.has) for n, w in readmission_cfg.windows.items()}
    original_nodes = set(readmission_cfg.window_nodes)

    relaxed = relax(
        readmission_cfg,
        {
            "remove_all_criteria": True,
            "collapse_temporal_gap_windows": True,
            "remove_post_label_windows": True,
        },
    )

    assert set(relaxed.window_nodes) == {"hospitalization.end", "target.end"}
    assert {n: dict(w.has) for n, w in readmission_cfg.windows.items()} == original_has
    assert set(readmission_cfg.window_nodes) == original_nodes
    assert readmission_cfg.window_tree.node_name == "trigger"
