"""Test set-up and fixtures code.

Provides sample ACES task configurations to the doctest namespace. Every fixture is function-scoped so that
relaxations (which mutate in place) never leak between tests.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from aces.config import (
    DerivedPredicateConfig,
    EventConfig,
    PlainPredicateConfig,
    TaskExtractorConfig,
    WindowConfig,
)

from aces_relaxations import RelaxedTaskConfig, print_ACES, strip_to_relevant_windows


def _hospital_predicates() -> dict[str, PlainPredicateConfig | DerivedPredicateConfig]:
    return {
        "admission": PlainPredicateConfig("ADMISSION"),
        "discharge": PlainPredicateConfig("DISCHARGE"),
        "death": PlainPredicateConfig("MEDS_DEATH"),
        "discharge_or_death": DerivedPredicateConfig("or(discharge, death)"),
    }


@pytest.fixture
def sample_MEDS_df() -> pl.DataFrame:
    """A tiny MEDS dataset matching ``sample_ACES_cfg``: one survivor, one death."""
    return pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2, 2],
            "time": [
                datetime(2020, 1, 1),
                datetime(2020, 1, 1, 12),
                datetime(2020, 1, 5),
                datetime(2020, 1, 1),
                datetime(2020, 1, 4),
            ],
            "code": ["ICU_ADMISSION", "LAB//A", "ICU_DISCHARGE", "ICU_ADMISSION", "MEDS_DEATH"],
            "numeric_value": [None, 1.0, None, None, None],
        }
    )


@pytest.fixture
def sample_ACES_cfg() -> TaskExtractorConfig:
    """A simple in-ICU mortality configuration: 24h input, 24h gap, label at next discharge or death."""
    return TaskExtractorConfig(
        predicates={
            "icu_admission": PlainPredicateConfig("ICU_ADMISSION"),
            "icu_discharge": PlainPredicateConfig("ICU_DISCHARGE"),
            "death": PlainPredicateConfig(code={"regex": "MEDS_DEATH.*"}),
            "discharge_or_death": DerivedPredicateConfig("or(icu_discharge, death)"),
        },
        trigger=EventConfig("icu_admission"),
        windows={
            "input": WindowConfig(
                "trigger",
                "start + 24h",
                False,
                True,
                index_timestamp="end",
                has={"icu_admission": "(None, 0)", "discharge_or_death": "(None, 0)"},
            ),
            "gap": WindowConfig(
                "input.end",
                "start + 24h",
                False,
                True,
                has={"icu_admission": "(None, 0)", "discharge_or_death": "(None, 0)"},
            ),
            "target": WindowConfig("gap.end", "start -> discharge_or_death", False, True, label="death"),
        },
    )


@pytest.fixture
def in_hosp_mortality_cfg() -> TaskExtractorConfig:
    """In-hospital mortality prediction with a sufficient-history requirement and a 24h gap window."""
    return TaskExtractorConfig(
        predicates=_hospital_predicates(),
        trigger=EventConfig("admission"),
        windows={
            "sufficient_history": WindowConfig(None, "trigger", True, False, has={"_ANY_EVENT": "(5, None)"}),
            "input": WindowConfig(
                "trigger",
                "start + 24h",
                False,
                True,
                index_timestamp="end",
                has={"admission": "(None, 0)", "discharge_or_death": "(None, 0)"},
            ),
            "gap": WindowConfig(
                "input.end",
                "start + 24h",
                False,
                True,
                has={"admission": "(None, 0)", "discharge_or_death": "(None, 0)"},
            ),
            "target": WindowConfig("gap.end", "start -> discharge_or_death", False, True, label="death"),
        },
    )


@pytest.fixture
def post_discharge_mortality_cfg() -> TaskExtractorConfig:
    """30-day post-discharge mortality prediction from the first 24h of an admission."""
    return TaskExtractorConfig(
        predicates=_hospital_predicates(),
        trigger=EventConfig("admission"),
        windows={
            "sufficient_history": WindowConfig(None, "trigger", True, False, has={"_ANY_EVENT": "(5, None)"}),
            "input": WindowConfig(
                "trigger",
                "start + 24h",
                False,
                True,
                index_timestamp="end",
                has={"admission": "(None, 0)", "discharge_or_death": "(None, 0)"},
            ),
            "post_input": WindowConfig(
                "input.end",
                "start + 1d",
                False,
                True,
                has={"admission": "(None, 0)", "discharge_or_death": "(None, 0)"},
            ),
            "hospitalization": WindowConfig(
                "input.end", "start -> discharge", False, True, has={"death": "(None, 0)"}
            ),
            "gap": WindowConfig(
                "hospitalization.end",
                "start + 1d",
                False,
                True,
                has={"admission": "(None, 0)", "death": "(None, 0)"},
            ),
            "target": WindowConfig("gap.end", "start + 29d", False, True, label="death"),
        },
    )


@pytest.fixture
def readmission_cfg() -> TaskExtractorConfig:
    """30-day readmission prediction at discharge, with a post-target censoring protection window."""
    return TaskExtractorConfig(
        predicates=_hospital_predicates(),
        trigger=EventConfig("discharge"),
        windows={
            "sufficient_history": WindowConfig(
                None, "hospitalization.start", True, False, has={"_ANY_EVENT": "(5, None)"}
            ),
            "hospitalization": WindowConfig(
                "end <- admission",
                "trigger",
                True,
                True,
                has={"_ANY_EVENT": "(10, None)"},
                index_timestamp="end",
            ),
            "gap": WindowConfig(
                "hospitalization.end",
                "start + 1d",
                False,
                True,
                has={"admission": "(None, 0)", "death": "(None, 0)"},
            ),
            "target": WindowConfig("gap.end", "start + 29d", False, True, label="admission"),
            "censoring_protection": WindowConfig(
                "target.end", None, True, True, has={"_ANY_EVENT": "(1, None)"}
            ),
        },
    )


@pytest.fixture
def two_stage_cfg() -> TaskExtractorConfig:
    """A two-stage infusion task: predict an adverse event during the second infusion stage."""
    return TaskExtractorConfig(
        predicates={
            "infusion_start": PlainPredicateConfig("INFUSION//START"),
            "infusion_end": PlainPredicateConfig("INFUSION//END"),
            "drug_given": PlainPredicateConfig("special_drug"),
            "adverse_event": PlainPredicateConfig("special_adverse_event"),
        },
        trigger=EventConfig("drug_given"),
        windows={
            "1st_infusion": WindowConfig(
                "trigger",
                "start -> infusion_end",
                True,
                True,
                has={"adverse_event": "(None, 0)"},
                index_timestamp="start",
            ),
            "2nd_infusion": WindowConfig(
                "1st_infusion.end -> infusion_start",
                "start -> infusion_end",
                True,
                True,
                label="adverse_event",
            ),
        },
    )


@pytest.fixture(autouse=True)
def _setup_doctest_namespace(
    doctest_namespace: dict[str, Any],
    sample_ACES_cfg: TaskExtractorConfig,
    sample_MEDS_df: pl.DataFrame,
    in_hosp_mortality_cfg: TaskExtractorConfig,
    post_discharge_mortality_cfg: TaskExtractorConfig,
    readmission_cfg: TaskExtractorConfig,
    two_stage_cfg: TaskExtractorConfig,
):
    doctest_namespace.update(
        {
            "datetime": datetime,
            "timedelta": timedelta,
            "tempfile": tempfile,
            "Path": Path,
            "print_ACES": print_ACES,
            "strip_to_relevant_windows": strip_to_relevant_windows,
            "RelaxedTaskConfig": RelaxedTaskConfig,
            "WindowConfig": WindowConfig,
            "sample_ACES_cfg": sample_ACES_cfg,
            "sample_MEDS_df": sample_MEDS_df,
            "pl": pl,
            "in_hosp_mortality_cfg": in_hosp_mortality_cfg,
            "post_discharge_mortality_cfg": post_discharge_mortality_cfg,
            "readmission_cfg": readmission_cfg,
            "two_stage_cfg": two_stage_cfg,
        }
    )
