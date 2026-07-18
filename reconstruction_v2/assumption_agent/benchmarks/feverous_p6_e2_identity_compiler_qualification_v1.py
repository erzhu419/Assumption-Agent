"""Auditable qualification of FEVEROUS lightweight identities vs compiler.

The qualification promotes only implementation evidence: a bounded real-DB
performance receipt and 64 content-independent real-page equivalence checks.
It never reads annotations, cohort/outcome data, or a selection secret.  Every
evidence file and executable source used by the diagnostic is content-hashed;
the later implementation freeze additionally binds these same paths as Git
blobs.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from assumption_agent.benchmarks import (
    feverous_p6_e2_identity_performance_diagnostic_v1 as diagnostic,
)


VERSION = "feverous_p6_e2_identity_compiler_qualification_v1"
SCHEMA = VERSION
STATUS = "passed_identity_full_compiler_equivalence_and_runtime_feasibility"
MANIFEST_RELATIVE = Path(
    "manifests/feverous_p6_e2_identity_compiler_qualification_v1.json"
)
PERFORMANCE_RECEIPT_RELATIVE = Path(
    "manifests/feverous_p6_e2_identity_performance_diagnostic_v1.json"
)
DIAGNOSTIC_SOURCE_RELATIVE = Path(
    "assumption_agent/benchmarks/"
    "feverous_p6_e2_identity_performance_diagnostic_v1.py"
)
DIAGNOSTIC_TEST_RELATIVE = Path(
    "tests/test_feverous_p6_e2_identity_performance_diagnostic_v1.py"
)
ATOMIC_SOURCE_RELATIVE = Path(
    "assumption_agent/benchmarks/feverous_atomic_corpus_v1.py"
)
ACQUISITION_SOURCE_RELATIVE = Path(
    "assumption_agent/benchmarks/feverous_p6_e2_acquisition_v1.py"
)
QUALIFICATION_SOURCE_RELATIVE = Path(
    "assumption_agent/benchmarks/"
    "feverous_p6_e2_identity_compiler_qualification_v1.py"
)
QUALIFICATION_TEST_RELATIVE = Path(
    "tests/test_feverous_p6_e2_identity_compiler_qualification_v1.py"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class FeverousIdentityCompilerQualificationError(RuntimeError):
    """The equivalence/performance evidence or a bound file drifted."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise FeverousIdentityCompilerQualificationError(
            "qualification is not canonical JSON"
        ) from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FeverousIdentityCompilerQualificationError(
            "qualification evidence file cannot be hashed"
        ) from exc
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FeverousIdentityCompilerQualificationError(
            "qualification evidence manifest is unavailable"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FeverousIdentityCompilerQualificationError(
            "qualification evidence manifest is invalid"
        ) from exc
    if not isinstance(value, dict):
        raise FeverousIdentityCompilerQualificationError(
            "qualification evidence must be an object"
        )
    return value


def _project(project: str | Path) -> Path:
    try:
        root = Path(project).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FeverousIdentityCompilerQualificationError(
            "qualification project root is unavailable"
        ) from exc
    if not root.is_dir():
        raise FeverousIdentityCompilerQualificationError(
            "qualification project root is not a directory"
        )
    return root


def _performance_evidence(project: Path) -> dict[str, Any]:
    path = project / PERFORMANCE_RECEIPT_RELATIVE
    receipt = _load_json(path)
    try:
        receipt_sha256 = diagnostic.verify_identity_performance_diagnostic_receipt(
            receipt
        )
    except diagnostic.FeverousIdentityPerformanceDiagnosticError as exc:
        raise FeverousIdentityCompilerQualificationError(
            "performance diagnostic receipt failed verification"
        ) from exc
    if (
        receipt.get("formal_valid") is not False
        or receipt.get("implementation_performance_only") is not True
        or receipt.get("observed_prefix_page_count", 0) < 100_000
        or receipt.get("real_crosscheck_sample_page_count") != 64
        or receipt.get("identity_full_compiler_mismatch_count") != 0
        or receipt.get("private_HMAC_heap_selection_included") is not False
        or receipt.get("claim_label_evidence_family_or_outcome_accessed")
        is not False
        or receipt.get("selection_secret_created_or_accessed") is not False
        or receipt.get("development_or_test_source_accessed") is not False
        or receipt.get("model_action_or_online_evaluator_calls") != 0
    ):
        raise FeverousIdentityCompilerQualificationError(
            "performance/equivalence evidence is insufficient"
        )
    bindings = {
        "performance_receipt_file_sha256": _sha256_file(path),
        "performance_diagnostic_receipt_sha256": receipt_sha256,
        "performance_diagnostic_source_file_sha256": _sha256_file(
            project / DIAGNOSTIC_SOURCE_RELATIVE
        ),
        "performance_diagnostic_test_file_sha256": _sha256_file(
            project / DIAGNOSTIC_TEST_RELATIVE
        ),
        "atomic_source_file_sha256": _sha256_file(
            project / ATOMIC_SOURCE_RELATIVE
        ),
        "acquisition_source_file_sha256": _sha256_file(
            project / ACQUISITION_SOURCE_RELATIVE
        ),
        "qualification_source_file_sha256": _sha256_file(
            project / QUALIFICATION_SOURCE_RELATIVE
        ),
        "qualification_test_file_sha256": _sha256_file(
            project / QUALIFICATION_TEST_RELATIVE
        ),
    }
    if (
        bindings["performance_diagnostic_source_file_sha256"]
        != receipt.get("diagnostic_source_sha256")
        or bindings["atomic_source_file_sha256"]
        != receipt.get("atomic_compiler_and_enumerator_source_sha256")
        or bindings["acquisition_source_file_sha256"]
        != receipt.get("acquisition_identity_source_sha256")
    ):
        raise FeverousIdentityCompilerQualificationError(
            "diagnostic executable source binding drifted"
        )
    return {"receipt": receipt, "bindings": bindings}


def form_identity_compiler_qualification(project: str | Path) -> dict[str, Any]:
    """Form, but do not write, the aggregate qualification manifest."""

    root = _project(project)
    evidence = _performance_evidence(root)
    receipt = evidence["receipt"]
    bindings = evidence["bindings"]
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": STATUS,
        "design_sha256": receipt["design_sha256"],
        "wikipedia_qualification_sha256": receipt[
            "wikipedia_qualification_sha256"
        ],
        "database_declared_sha256": receipt["database_declared_sha256"],
        "database_expected_row_count": receipt["database_expected_row_count"],
        "identity_enumerator_version": receipt["identity_enumerator_version"],
        "atomic_compiler_version": receipt["atomic_compiler_version"],
        "real_sample_policy": receipt["real_crosscheck_sample_policy"],
        "real_sample_page_count": receipt["real_crosscheck_sample_page_count"],
        "real_sample_eligible_identity_count": receipt[
            "real_crosscheck_eligible_identity_count"
        ],
        "real_sample_excluded_empty_count": receipt[
            "real_crosscheck_excluded_empty_count"
        ],
        "real_sample_aggregate_sha256": receipt[
            "real_crosscheck_aggregate_sha256"
        ],
        "identity_full_compiler_mismatch_count": receipt[
            "identity_full_compiler_mismatch_count"
        ],
        "runtime_prefix_page_count": receipt["observed_prefix_page_count"],
        "runtime_prefix_payload_utf8_bytes": receipt[
            "observed_prefix_payload_utf8_bytes"
        ],
        "runtime_prefix_eligible_identity_count": receipt[
            "observed_prefix_eligible_identity_count"
        ],
        "runtime_prefix_wall_seconds": receipt["observed_prefix_wall_seconds"],
        "runtime_prefix_peak_rss_growth_kib": receipt[
            "process_peak_rss_growth_kib"
        ],
        "projected_complete_identity_scan_seconds": receipt[
            "projected_complete_identity_scan_seconds"
        ],
        "projected_complete_identity_scan_hours": receipt[
            "projected_complete_identity_scan_hours"
        ],
        "projection_scope": receipt["projection_scope"],
        "private_HMAC_heap_selection_included": False,
        "evidence_bindings": {
            "performance_receipt": {
                "relative_path": PERFORMANCE_RECEIPT_RELATIVE.as_posix(),
                "file_sha256": bindings["performance_receipt_file_sha256"],
                "receipt_sha256": bindings[
                    "performance_diagnostic_receipt_sha256"
                ],
            },
            "performance_diagnostic_source": {
                "relative_path": DIAGNOSTIC_SOURCE_RELATIVE.as_posix(),
                "file_sha256": bindings[
                    "performance_diagnostic_source_file_sha256"
                ],
            },
            "performance_diagnostic_test": {
                "relative_path": DIAGNOSTIC_TEST_RELATIVE.as_posix(),
                "file_sha256": bindings[
                    "performance_diagnostic_test_file_sha256"
                ],
            },
            "atomic_source": {
                "relative_path": ATOMIC_SOURCE_RELATIVE.as_posix(),
                "file_sha256": bindings["atomic_source_file_sha256"],
            },
            "acquisition_source": {
                "relative_path": ACQUISITION_SOURCE_RELATIVE.as_posix(),
                "file_sha256": bindings["acquisition_source_file_sha256"],
            },
            "qualification_source": {
                "relative_path": QUALIFICATION_SOURCE_RELATIVE.as_posix(),
                "file_sha256": bindings["qualification_source_file_sha256"],
            },
            "qualification_test": {
                "relative_path": QUALIFICATION_TEST_RELATIVE.as_posix(),
                "file_sha256": bindings["qualification_test_file_sha256"],
            },
        },
        "qualification_interpretation": (
            "lightweight_identity_equals_full_compiler_on_64_frozen_real_pages_"
            "and_100k_page_identity_runtime_is_feasible_but_private_HMAC_cost_"
            "remains_separate"
        ),
        "formal_valid_acquisition_performed": False,
        "annotation_claim_label_evidence_or_family_accessed": False,
        "selection_cohort_or_outcome_accessed": False,
        "formal_selection_secret_generated_or_accessed": False,
        "development_or_test_source_accessed": False,
        "online_evaluator_calls": 0,
    }
    return {**body, "qualification_sha256": stable_hash(body)}


def verify_identity_compiler_qualification(
    project: str | Path,
) -> Mapping[str, Any]:
    root = _project(project)
    qualification = _load_json(root / MANIFEST_RELATIVE)
    body = dict(qualification)
    declared = body.pop("qualification_sha256", None)
    if (
        not isinstance(declared, str)
        or _SHA256.fullmatch(declared) is None
        or stable_hash(body) != declared
        or qualification.get("schema") != SCHEMA
        or qualification.get("version") != VERSION
        or qualification.get("status") != STATUS
        or qualification.get("real_sample_page_count") != 64
        or qualification.get("identity_full_compiler_mismatch_count") != 0
        or qualification.get("runtime_prefix_page_count", 0) < 100_000
        or qualification.get("private_HMAC_heap_selection_included") is not False
        or qualification.get("formal_valid_acquisition_performed") is not False
        or qualification.get("annotation_claim_label_evidence_or_family_accessed")
        is not False
        or qualification.get("selection_cohort_or_outcome_accessed") is not False
        or qualification.get("formal_selection_secret_generated_or_accessed")
        is not False
        or qualification.get("development_or_test_source_accessed") is not False
        or qualification.get("online_evaluator_calls") != 0
    ):
        raise FeverousIdentityCompilerQualificationError(
            "identity/compiler qualification semantics drifted"
        )
    expected = form_identity_compiler_qualification(root)
    if qualification != expected:
        raise FeverousIdentityCompilerQualificationError(
            "identity/compiler qualification evidence binding drifted"
        )
    return qualification


__all__ = [
    "FeverousIdentityCompilerQualificationError",
    "form_identity_compiler_qualification",
    "verify_identity_compiler_qualification",
]
