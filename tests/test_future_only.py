"""Tests for the future-only validation error message."""

from contextlib import redirect_stdout
from io import StringIO

import pytest
from aces.config import EventConfig, PlainPredicateConfig, TaskExtractorConfig, WindowConfig
from bigtree import print_tree

from aces_relaxations import validate_future_only


def test_validate_future_only_error_includes_tree_when_prediction_not_ancestor():
    """Ensures the validation error message contains the window tree when misconfigured."""
    task_cfg = TaskExtractorConfig(
        predicates={
            "trigger_pred": PlainPredicateConfig("TRIGGER"),
            "label_pred": PlainPredicateConfig("LABEL"),
        },
        trigger=EventConfig("trigger_pred"),
        windows={
            "prediction": WindowConfig(
                start="trigger + 1h",
                end="start + 1h",
                start_inclusive=True,
                end_inclusive=True,
                has={},
                index_timestamp="start",
            ),
            "label": WindowConfig(
                start="end - 1h",
                end="trigger + 4h",
                start_inclusive=False,
                end_inclusive=True,
                has={},
                label="label_pred",
            ),
        },
    )

    with pytest.raises(ValueError) as excinfo:
        validate_future_only(task_cfg)

    strio = StringIO()
    with redirect_stdout(strio):
        print_tree(task_cfg.window_tree)
    tree_str = strio.getvalue()

    message = str(excinfo.value)
    assert "prediction time window node (prediction.start)" in message
    assert tree_str in message
