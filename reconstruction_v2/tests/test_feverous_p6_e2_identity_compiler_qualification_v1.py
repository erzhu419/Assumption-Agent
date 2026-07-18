from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest

from assumption_agent.benchmarks import (
    feverous_p6_e2_identity_compiler_qualification_v1 as module,
)


PROJECT = Path(__file__).resolve().parents[1]


def _fixture_project(tmp_path: Path) -> Path:
    paths = (
        module.PERFORMANCE_RECEIPT_RELATIVE,
        module.DIAGNOSTIC_SOURCE_RELATIVE,
        module.DIAGNOSTIC_TEST_RELATIVE,
        module.ATOMIC_SOURCE_RELATIVE,
        module.ACQUISITION_SOURCE_RELATIVE,
        module.QUALIFICATION_SOURCE_RELATIVE,
        module.QUALIFICATION_TEST_RELATIVE,
    )
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / relative, target)
    return tmp_path


def test_forms_and_verifies_exact_dependency_bound_qualification(
    tmp_path: Path,
) -> None:
    project = _fixture_project(tmp_path)
    qualification = module.form_identity_compiler_qualification(project)
    assert qualification["status"] == (
        "passed_identity_full_compiler_equivalence_and_runtime_feasibility"
    )
    assert qualification["real_sample_page_count"] == 64
    assert qualification["identity_full_compiler_mismatch_count"] == 0
    assert qualification["runtime_prefix_page_count"] == 100_000
    assert qualification["selection_cohort_or_outcome_accessed"] is False
    manifest = project / module.MANIFEST_RELATIVE
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert module.verify_identity_compiler_qualification(project) == qualification

    tampered = copy.deepcopy(qualification)
    tampered["runtime_prefix_page_count"] = 100_001
    manifest.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        module.FeverousIdentityCompilerQualificationError,
        match="semantics drifted",
    ):
        module.verify_identity_compiler_qualification(project)


def test_dependency_receipt_forgery_is_rejected(tmp_path: Path) -> None:
    project = _fixture_project(tmp_path)
    receipt_path = project / module.PERFORMANCE_RECEIPT_RELATIVE
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["real_crosscheck_sample_page_count"] = 63
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(
        module.FeverousIdentityCompilerQualificationError,
        match="failed verification",
    ):
        module.form_identity_compiler_qualification(project)
