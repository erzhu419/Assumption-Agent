from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
ARTIFACT_PATH = PROJECT_ROOT / "artifacts/phase3_shrink2_dual_qualification_v1.json"
REPORT_ID_DOMAIN = b"HEGEL/SHRINK2/DUAL_QUALIFICATION/V1\0"
FILE_SET_DOMAIN = b"HEGEL/SHRINK2/COMMIT_F_FILE_SET/V1\0"


def _git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=check,
        capture_output=True,
    )


def test_shrink2_qualification_artifact_binds_exact_commit_sources() -> None:
    report = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    binding = report["repository_binding"]
    commit = binding["commit"]

    assert _git("rev-parse", f"{commit}^{{tree}}").stdout.decode().strip() == binding[
        "commit_tree"
    ]
    assert _git("show", "-s", "--format=%s", commit).stdout.decode().strip() == binding[
        "commit_subject"
    ]

    file_set_hasher = hashlib.sha256(FILE_SET_DOMAIN)
    source_rows = sorted(binding["source_files"], key=lambda row: row["path"])
    assert len(source_rows) == 14
    for row in source_rows:
        path = row["path"]
        payload = _git("show", f"{commit}:Hegel Machine/{path}").stdout
        digest = hashlib.sha256(payload).digest()
        assert "sha256:" + digest.hex() == row["sha256"]
        encoded_path = path.encode("utf-8")
        file_set_hasher.update(struct.pack(">Q", len(encoded_path)))
        file_set_hasher.update(encoded_path)
        file_set_hasher.update(digest)
    assert "sha256:" + file_set_hasher.hexdigest() == binding[
        "source_file_set_root"
    ]

    for path in report["excluded_uncommitted_diagnostic_paths"]:
        probe = _git("cat-file", "-e", f"{commit}:Hegel Machine/{path}", check=False)
        assert probe.returncode != 0


def test_shrink2_qualification_report_id_and_claim_boundary_are_exact() -> None:
    report = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    report_id = report["qualification_report_id"]
    report["qualification_report_id"] = None
    canonical_diagnostic_json = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    expected = "phase3_shrink2_dual_qualification_" + hashlib.sha256(
        REPORT_ID_DOMAIN + canonical_diagnostic_json
    ).hexdigest()
    assert report_id == expected

    assert report["status"] == "VERIFIED_DIAGNOSTIC_ONLY"
    assert report["execution_state"] == "NOT_RUN"
    assert report["executed_closure_status"] == "NOT_RUN"
    assert report["formal_roots"] is None
    assert report["formal_archive_roots_generated"] is False
    assert report["complete_closure_enumerated"] is False
    assert report["complete_claim_allowed"] is False
    assert report["outside_certificate_issued"] is False
    assert report["active_promotion_allowed"] is False
    assert report["target_role_evaluation_performed"] is False
    assert report["seed_material"] is None
    assert report["signature_bundle"] is None


def test_shrink2_dual_qualification_counts_and_commitments_are_consistent() -> None:
    report = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    golden = report["dual_golden_replay"]
    capacity = report["dual_capacity_replay"]
    differential = report["differential_qualification"]
    regression = report["regression_tests"]

    assert golden["cross_language_equal"] is True
    assert golden["python_passed_count"] == golden["rust_passed_count"] == 59
    assert capacity["cross_language_equal"] is True
    assert capacity["accepted_source_count"] == 2160
    assert capacity["accepted_unique_count"] == 2160
    assert capacity["rejected_count"] == 0
    assert capacity["subset_status"] == "SUBSET_ONLY_NOT_COMPLETE"
    assert capacity["accepted_set_commitment"] == (
        "sha256:9045e4ebe6416dcbf699e7972f25468aef45c0f0aec0e58806061b7ce64d790e"
    )

    primary = differential["primary_adversarial_audit"]
    assert primary["total_cases"] == sum(
        primary[name]
        for name in (
            "arbitrary_10000_digit_cases",
            "curated_source_cases",
            "exhaustive_source_cases",
            "formal_cases",
            "lexical_numeric_cases",
            "seeded_source_cases",
        )
    )
    independent = differential["independent_audit"]
    assert independent["total_cases"] == sum(
        independent[name]
        for name in (
            "formal_ast_cases",
            "malformed_multi_error_source_cases",
            "raw_cbor_cases",
            "source_normalization_and_boundary_cases",
        )
    )
    assert primary["mismatches"] == independent["mismatches"] == 0
    assert differential["panic_count"] == 0
    assert differential["accepted_program_cbor_and_hash_equal"] is True

    assert regression == {
        "python_combined_collected": 165,
        "python_combined_passed": 162,
        "python_combined_skipped": 3,
        "python_focused_passed": 100,
        "rust_integration_passed": 3,
        "rust_unit_passed": 17,
    }
