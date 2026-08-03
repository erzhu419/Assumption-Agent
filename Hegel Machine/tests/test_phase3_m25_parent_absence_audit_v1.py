from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import pytest

from hegel_machine.phase3_m25_parent_absence_audit_v1 import (
    CONTENT_PREDICATE_PROFILE_ID,
    PARENT_DSL_VERSION,
    PARENT_FREEZE_VERSION,
    ParentAbsenceAuditEvidence,
    ParentAbsenceAuditError,
    _CONTENT_ABSENCE_PREDICATES,
    _batch_blob_sizes,
    _blob_inventory_digest,
    _batch_cat_file,
    _commit_touched_refs,
    _digest_set_sha256,
    _legacy_source_rows,
    _parse_commit_parents,
    _path_name_receipt,
    _row_for_blob,
    build_parent_absence_attestation_fields_v2,
    generate_parent_absence_audit_v1,
    parent_absence_public_receipt_v1,
    replay_parent_absence_audit_v1,
)
from hegel_machine.phase3_m25_wire_v1 import (
    AUDITED_PARENT_COMMIT_SHA1,
    candidate_content_root,
    candidate_record_tree_root,
    git_sha1_commit_id,
    id_digest_v1,
)


def _git(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    return completed.stdout


def _write(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def _commit(repository: Path, message: str) -> bytes:
    _git(repository, "add", "-A")
    _git(repository, "commit", "-m", message)
    return bytes.fromhex(_git(repository, "rev-parse", "HEAD").strip().decode("ascii"))


def _merge_fixture(tmp_path: Path) -> tuple[Path, bytes, tuple[bytes, bytes], dict[str, bytes]]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q", "-b", "left")
    _git(repository, "config", "user.name", "Gate 17 Test")
    _git(repository, "config", "user.email", "gate17@example.invalid")

    _write(repository / "gone.txt", "root\n")
    _write(repository / "shared.txt", "root\n")
    root = _commit(repository, "root")
    _git(repository, "branch", "right")

    _write(repository / "gone.txt", "left\n")
    _write(repository / "shared.txt", "left\n")
    left = _commit(repository, "left")

    _git(repository, "checkout", "-q", "right")
    _write(repository / "gone.txt", "right\n")
    _write(repository / "shared.txt", "right\n")
    right = _commit(repository, "right")

    _git(repository, "checkout", "-q", "left")
    completed = subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", "right"],
        cwd=repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode != 0  # both files conflict by construction
    (repository / "gone.txt").unlink()
    _write(repository / "shared.txt", "merge\n")
    merge = _commit(repository, "merge")

    digests = {
        "gone_left": bytes.fromhex(
            _git(repository, "rev-parse", f"{left.hex()}:gone.txt").strip().decode("ascii")
        ),
        "gone_right": bytes.fromhex(
            _git(repository, "rev-parse", f"{right.hex()}:gone.txt").strip().decode("ascii")
        ),
        "shared_merge": bytes.fromhex(
            _git(repository, "rev-parse", f"{merge.hex()}:shared.txt").strip().decode("ascii")
        ),
    }
    del root
    return repository, merge, (left, right), digests


def _synthetic_evidence() -> ParentAbsenceAuditEvidence:
    path_row = {
        "repository_path_alias_id_digest": id_digest_v1("repo-path:gate17-test"),
        "raw_repository_path_utf8_bytes": b"Hegel Machine/legacy/source.json",
        "git_object_algorithm_id": 1,
        "git_blob_digest": bytes.fromhex("11" * 20),
        "file_mode": 0o100644,
        "byte_length": 7,
    }
    touched_root = candidate_record_tree_root("AuditedPathBlobRecordV1", [path_row])
    history_row = {
        "commit_generation": 0,
        "repository_commit_id": git_sha1_commit_id(AUDITED_PARENT_COMMIT_SHA1),
        "ordered_parent_commit_ids": (),
        "touched_path_set_root": touched_root,
    }
    legacy_rows = _legacy_source_rows()
    bundle = {
        "audited_parent_repository_commit_id": git_sha1_commit_id(
            AUDITED_PARENT_COMMIT_SHA1
        ),
        "audited_path_tree_root": touched_root,
        "audited_history_tree_root": candidate_record_tree_root(
            "AuditedHistoryRowV1", [history_row]
        ),
        "legacy_source_tree_root": candidate_record_tree_root(
            "LegacyParentSourceRowV1", legacy_rows
        ),
        "audited_path_count": 1,
        "audited_history_row_count": 1,
        "legacy_source_count": 2,
    }
    audit_root = candidate_content_root("ParentAbsenceAuditBundleV1", bundle)
    static = {
        "parent_dsl_version_digest": id_digest_v1(PARENT_DSL_VERSION),
        "parent_freeze_version_digest": id_digest_v1(PARENT_FREEZE_VERSION),
        "parent_repository_commit_id": git_sha1_commit_id(AUDITED_PARENT_COMMIT_SHA1),
        "audit_bundle_root": audit_root,
        "absence_reason_bitmask": 0b1111,
    }
    content_audit = {
        "content_predicate_profile_id": CONTENT_PREDICATE_PROFILE_ID,
        "inspected_path_blob_row_count": 1,
        "inspected_unique_blob_count": 1,
        "inspected_total_byte_length": 7,
        "inspected_blob_inventory_sha256": _blob_inventory_digest(
            {path_row["git_blob_digest"]: 7}
        ),
        "git_blob_object_id_and_size_verified": True,
        "structured_candidate_unique_blob_count": 1,
        "unscannable_relevant_structured_blob_count": 0,
        "content_absence_predicates": [
            {
                "predicate_id": predicate_id,
                "exact_signatures_ascii": [
                    signature.decode("ascii") for signature in signatures
                ],
                "match_occurrence_count": 0,
                "matching_unique_blob_count": 0,
                "matching_path_blob_row_count": 0,
                "matching_blob_digest_set_sha256": _digest_set_sha256(set()),
                "absent": True,
            }
            for predicate_id, signatures in _CONTENT_ABSENCE_PREDICATES.items()
        ],
        "legacy_source_presence": [
            {
                "legacy_parent_payload_source_id": source_id,
                "match_occurrence_count": 1,
                "matching_unique_blob_count": 1,
                "matching_path_blob_row_count": 1,
                "matching_blob_digest_set_sha256": _digest_set_sha256(
                    {path_row["git_blob_digest"]}
                ),
                "present": True,
            }
            for source_id in (
                "target_spec_b491c0a9719fb0279fe02798ede026e440c17a539965514145a7818b15387ac3",
                "sink_control_spec_7fd6f9a6e2b4c6eda0c7e1545ad42cb19666743ede8ed87f40d82c0ef46198a0",
            )
        ],
        "all_content_absence_predicates_absent": True,
        "all_legacy_sources_present": True,
    }
    return ParentAbsenceAuditEvidence(
        top_level_path_rows=(path_row,),
        history_rows=(history_row,),
        touched_path_rows_by_history_row=((path_row,),),
        legacy_source_rows=legacy_rows,
        audit_bundle_fields=bundle,
        audit_bundle_root=audit_root,
        attestation_static_fields=static,
        path_name_receipt=_path_name_receipt(
            [path_row],
            audit_bundle_root=audit_root,
            audited_path_tree_root=touched_root,
            content_blob_audit=content_audit,
        ),
    )


def test_real_parent_versions_are_not_the_old_synthetic_fixture_value() -> None:
    assert PARENT_DSL_VERSION == "hegel-old-dsl-v1.0.0"
    assert PARENT_FREEZE_VERSION == "hegel-freeze-p2b-p3-v1.0.2"
    assert id_digest_v1(PARENT_FREEZE_VERSION) != id_digest_v1(
        "hegel-freeze-p2b-p3-v1.0.0"
    )


def test_any_parent_rule_preserves_two_deleted_parent_blobs_and_one_result_blob(
    tmp_path: Path,
) -> None:
    repository, merge, expected_parents, digests = _merge_fixture(tmp_path)
    object_type, commit_payload = _batch_cat_file(repository, [merge])[merge]
    assert object_type == b"commit"
    parents = _parse_commit_parents(merge, commit_payload)
    assert parents == expected_parents  # commit-object parent order, not a sorted set

    refs = _commit_touched_refs(repository, merge, parents)
    by_path: dict[bytes, list[bytes]] = {}
    for ref in refs:
        by_path.setdefault(ref.path, []).append(ref.digest)
    assert set(by_path[b"gone.txt"]) == {digests["gone_left"], digests["gone_right"]}
    assert by_path[b"shared.txt"] == [digests["shared_merge"]]
    assert len(refs) == 3

    sizes = _batch_blob_sizes(repository, [ref.digest for ref in refs])
    rows = sorted((_row_for_blob(ref, sizes) for ref in refs), key=lambda row: (
        row["raw_repository_path_utf8_bytes"],
        row["repository_path_alias_id_digest"],
        row["git_blob_digest"],
    ))
    assert len(candidate_record_tree_root("AuditedPathBlobRecordV1", rows)) == 32


def test_structural_replay_and_unsigned_attestation_completion() -> None:
    evidence = _synthetic_evidence()
    assert replay_parent_absence_audit_v1(evidence) == evidence.audit_bundle_root
    fields = build_parent_absence_attestation_fields_v2(
        evidence,
        auditor_key_id=bytes.fromhex("22" * 16),
        audited_at_unix_seconds=1_800_000_000,
    )
    assert fields["parent_freeze_version_digest"] == id_digest_v1(
        "hegel-freeze-p2b-p3-v1.0.2"
    )
    assert fields["auditor_key_id"] == bytes.fromhex("22" * 16)
    assert "signature" not in fields


def test_replay_rejects_wrong_generation_and_static_parent_freeze() -> None:
    evidence = _synthetic_evidence()
    bad_history = (dict(evidence.history_rows[0], commit_generation=1),)
    with pytest.raises(Exception, match="FAIL_PARENT_AUDIT_HISTORY"):
        replay_parent_absence_audit_v1(replace(evidence, history_rows=bad_history))

    bad_static = dict(evidence.attestation_static_fields)
    bad_static["parent_freeze_version_digest"] = id_digest_v1(
        "hegel-freeze-p2b-p3-v1.0.0"
    )
    with pytest.raises(ParentAbsenceAuditError) as caught:
        replay_parent_absence_audit_v1(
            replace(evidence, attestation_static_fields=bad_static)
        )
    assert caught.value.code == "FAIL_PARENT_AUDIT_REPLAY_MISMATCH"


def test_replay_rejects_non_union_top_level_rows_and_receipt_tamper() -> None:
    evidence = _synthetic_evidence()
    with pytest.raises(Exception, match="FAIL_PARENT_AUDIT_PATH_UNION"):
        replay_parent_absence_audit_v1(
            replace(evidence, top_level_path_rows=())
        )

    bad_receipt = dict(evidence.path_name_receipt)
    bad_receipt["all_predicates_absent"] = False
    with pytest.raises(ParentAbsenceAuditError) as caught:
        replay_parent_absence_audit_v1(
            replace(evidence, path_name_receipt=bad_receipt)
        )
    assert caught.value.code == "FAIL_PARENT_AUDIT_REPLAY_MISMATCH"

    bad_content_receipt = dict(evidence.path_name_receipt)
    bad_content = dict(bad_content_receipt["content_blob_audit"])
    bad_absence = [dict(row) for row in bad_content["content_absence_predicates"]]
    bad_absence[0]["match_occurrence_count"] = 1
    bad_absence[0]["absent"] = False
    bad_content["content_absence_predicates"] = bad_absence
    bad_content["all_content_absence_predicates_absent"] = False
    bad_content_receipt["content_blob_audit"] = bad_content
    with pytest.raises(ParentAbsenceAuditError) as caught:
        replay_parent_absence_audit_v1(
            replace(evidence, path_name_receipt=bad_content_receipt)
        )
    assert caught.value.code == "FAIL_PARENT_AUDIT_REPLAY_MISMATCH"


@pytest.mark.skipif(
    os.environ.get("HEGEL_RUN_FULL_PARENT_AUDIT") != "1",
    reason="set HEGEL_RUN_FULL_PARENT_AUDIT=1 for the 1,399-commit Git-object replay",
)
def test_frozen_parent_full_history_integration() -> None:
    repository = Path(__file__).resolve().parents[2]
    evidence = generate_parent_absence_audit_v1(repository)
    assert replay_parent_absence_audit_v1(evidence, repository=repository) == bytes.fromhex(
        "136c9eee4c616d9f55dae699cb467e56921ce4706943ae87a5ad89bf9d82ff51"
    )
    receipt = parent_absence_public_receipt_v1(evidence)
    assert receipt["audited_history_row_count"] == 1399
    assert receipt["audited_path_count"] == 7945
    assert receipt["root_commit_count"] == 1
    assert receipt["merge_commit_count"] == 1
    assert receipt["all_predicates_absent"] is True
    assert receipt["content_blob_audit"]["inspected_unique_blob_count"] == 7792
    assert receipt["content_blob_audit"]["inspected_total_byte_length"] == 1_520_624_571
    assert receipt["content_blob_audit"]["inspected_blob_inventory_sha256"] == (
        "b30255e89dc306a07ae72d25c1b67c86fe6552a8d69268e6d4d8a3008ebcdd09"
    )
    assert receipt["content_blob_audit"][
        "all_content_absence_predicates_absent"
    ] is True
    assert receipt["content_blob_audit"]["all_legacy_sources_present"] is True
    assert receipt["audited_path_tree_root"] == (
        "55c4670498efcfb80055f6a0ada0c3b44da2f24c82a1268701a38834a649cc3f"
    )
    assert receipt["audited_history_tree_root"] == (
        "c8b59bf44f5020656f34932c3e0394959d26e0438bf75f0040ac93f449077854"
    )
    assert receipt["legacy_source_tree_root"] == (
        "982a60f88ceee5a08f3f0ab4cb44002308ce4b288de334407e02fdc210bbf3c7"
    )
    artifact_path = repository / (
        "Hegel Machine/artifacts/phase3_m25_parent_absence_audit_receipt_v1.json"
    )
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == receipt
