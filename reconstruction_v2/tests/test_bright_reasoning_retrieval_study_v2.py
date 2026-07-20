from __future__ import annotations

import pytest

from assumption_agent.benchmarks import bright_reasoning_retrieval_study_v1 as v1
from assumption_agent.benchmarks import bright_reasoning_retrieval_study_v2 as v2


def test_only_u0000_is_replaced_before_frozen_truncation() -> None:
    assert v2.normalize_document_content("alpha\x00beta") == "alpha\ufffdbeta"
    long = "x" * (v1.DOCUMENT_TEXT_CHARACTERS + 10)
    assert v2.normalize_document_content(long) == long[: v1.DOCUMENT_TEXT_CHARACTERS]
    with pytest.raises(v1.BrightStudyError, match="content"):
        v2.normalize_document_content("   ")


def test_v2_activation_redirects_paths_and_preserves_study_constants() -> None:
    names = (
        "VERSION",
        "DESIGN_SCHEMA",
        "FREEZE_SCHEMA",
        "CORPUS_RESULT_SCHEMA",
        "STAGE_RESULT_SCHEMA",
        "ACTION_SCHEMA",
        "SCORED_SCHEMA",
        "MARKER_SCHEMA",
        "DESIGN_RELATIVE",
        "FREEZE_RELATIVE",
        "DESIGN_SELF_SHA256",
        "DESIGN_FILE_SHA256",
        "FORMAL_ROOT_RELATIVE",
        "CORPUS_RESULT_RELATIVE",
        "G_RESULT_RELATIVE",
        "A_FORM_RESULT_RELATIVE",
        "F_RESULT_RELATIVE",
        "A_HOLD_RESULT_RELATIVE",
        "M_RESULT_RELATIVE",
        "PUBLIC_STAGE_RESULTS",
        "STAGE_PREDECESSORS",
        "_read_source_documents",
    )
    original = {
        name: getattr(v1, name) for name in names
    }
    try:
        v2._activate_v2()
        assert v1.VERSION == v2.VERSION
        assert v1.FORMAL_ROOT_RELATIVE == v2.FORMAL_ROOT_RELATIVE
        assert v1.PUBLIC_STAGE_RESULTS["M_search"] == v2.M_RESULT_RELATIVE
        assert v1.core.RECIPE_ORDER == (
            "P1_RRF_EQUAL",
            "P2_RRF_ANCHOR2",
            "P3_MAX_SIM",
            "P4_MEAN_SIM",
            "P5_TOP2_MEAN",
            "P6_RELATION_MECHANISM_RRF",
            "P7_ENTITY_CONSTRAINT_RRF",
            "P8_ROUND_ROBIN",
        )
    finally:
        for name, value in original.items():
            setattr(v1, name, value)
