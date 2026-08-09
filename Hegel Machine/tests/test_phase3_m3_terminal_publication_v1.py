from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import hegel_machine.phase3_m3_terminal_publication_v1 as publication


def _load() -> dict[str, object]:
    return json.loads(publication.ARTIFACT_PATH.read_bytes())


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _refresh_self_hash(document: dict[str, object]) -> None:
    body = dict(document)
    body.pop("outcome_artifact_sha256", None)
    document["outcome_artifact_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()


def test_real_terminal_publication_validates_with_narrow_claim_boundary() -> None:
    summary = publication.validate()

    assert summary["artifact_file_sha256"] == (
        "4f631224383297f6f30d70dbcefc15ed1c1296ba634a604e5a59562d11e67aed"
    )
    assert summary["outcome_self_sha256"] == (
        "973214b278e0bd3af474fa0b095e518e1ea8323917845b856e8b5de72913c67c"
    )
    assert summary["closure_status"] == "DSL_TOO_LARGE"
    assert summary["canonical_program_count"] == 50_000
    assert summary["first_out_of_budget_program_hash_hex"] == (
        "96200a6a131204315ffcd1efd0aa2dcfe2ce665a2c06516461772c9812f0ec71"
    )
    assert summary["claim_boundary"] == {
        "bounded_child_dsl_dsl_too_large": True,
        "complete_closure": False,
        "outside_frozen_closure_certificate": False,
        "phase3_exit": False,
        "active_promotion": False,
    }
    assert summary["next_action"] == (
        "SHRINK_STEP_2_REDUCE_RATIONAL_PARAMETER_TO_NEG1_ZERO_POS1"
    )


def test_three_file_publication_carrier_binds_exact_start_and_terminal() -> None:
    summary = publication.validate_publication_carrier_v1()

    assert summary["schema"] == "hegel-phase3-m3-terminal-publication-carrier/1"
    assert summary["start_state_record_root_hex"] == (
        "daa8341296a6fc075346a0bb6df95667eb726dd119f212437c2b2e645e0d91e0"
    )
    assert summary["carrier_files"] == {
        publication.START_STATE_REPOSITORY_PATH: {
            "byte_length": 1_525,
            "sha256": "9f07564d4f859e082288ddf971c336a03b490062c65bce7eb81ddcfa64ea4053",
        },
        publication.START_PUBLICATION_RECEIPT_REPOSITORY_PATH: {
            "byte_length": 26_879,
            "sha256": "dede9fb1bf1febe4ec6646f00be456c94ff181fa91e23a23d8392c7596a70df3",
        },
        publication.ARTIFACT_REPOSITORY_PATH: {
            "byte_length": 62_942,
            "sha256": "4f631224383297f6f30d70dbcefc15ed1c1296ba634a604e5a59562d11e67aed",
        },
    }


def test_publication_carrier_rejects_tampered_start_state(tmp_path: Path) -> None:
    for source in (
        publication.START_STATE_PATH,
        publication.START_PUBLICATION_RECEIPT_PATH,
        publication.ARTIFACT_PATH,
    ):
        (tmp_path / source.name).write_bytes(source.read_bytes())
    state_path = tmp_path / publication.START_STATE_PATH.name
    state = json.loads(state_path.read_bytes())
    state["formal_gate_count"] = 23
    state_path.write_bytes(_canonical(state))

    with pytest.raises(publication.M3TerminalPublicationError) as caught:
        publication.validate_publication_carrier_v1(tmp_path)

    assert caught.value.code == publication.FAIL_FILE_HASH


def test_real_artifact_is_exact_canonical_compact_json_plus_newline() -> None:
    payload = publication.ARTIFACT_PATH.read_bytes()

    assert len(payload) == publication.EXPECTED_ARTIFACT_BYTE_LENGTH
    assert hashlib.sha256(payload).hexdigest() == (
        publication.EXPECTED_ARTIFACT_FILE_SHA256
    )
    assert payload == _canonical(json.loads(payload))
    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")


def test_duplicate_json_key_fails_closed_before_hash_acceptance(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_bytes(b'{"schema":"one","schema":"two"}\n')

    with pytest.raises(publication.M3TerminalPublicationError) as caught:
        publication.validate(path)

    assert caught.value.code == publication.FAIL_JSON


def test_noncanonical_json_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "pretty.json"
    path.write_text(json.dumps(_load(), indent=2, sort_keys=True) + "\n", encoding="ascii")

    with pytest.raises(publication.M3TerminalPublicationError) as caught:
        publication.validate(path)

    assert caught.value.code == publication.FAIL_CANONICAL


def test_canonical_but_modified_file_fails_exact_file_hash(tmp_path: Path) -> None:
    document = deepcopy(_load())
    document["runner_evidence_files"][0]["byte_length"] += 1
    _refresh_self_hash(document)
    path = tmp_path / "modified.json"
    path.write_bytes(_canonical(document))

    with pytest.raises(publication.M3TerminalPublicationError) as caught:
        publication.validate(path)

    assert caught.value.code == publication.FAIL_FILE_HASH


def test_outcome_self_hash_tamper_fails_closed() -> None:
    document = deepcopy(_load())
    document["canonical_program_count"] = 49_999

    with pytest.raises(publication.M3TerminalPublicationError) as caught:
        publication.build_publication_summary(document)

    assert caught.value.code == publication.FAIL_SELF_HASH


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("run_id_hex", "0" * 32),
        ("publication_commit_b", "0" * 40),
        ("execution_manifest_root_hex", "0" * 64),
        ("closure_status", "COMPLETE"),
        ("canonical_program_count", 49_999),
        ("raw_operator_application_count", 3_292_438),
        ("first_out_of_budget_program_hash_hex_or_null", "0" * 64),
    ),
)
def test_fixed_identity_tamper_with_refreshed_self_hash_fails_closed(
    field: str, value: object
) -> None:
    document = deepcopy(_load())
    document[field] = value
    _refresh_self_hash(document)

    with pytest.raises(publication.M3TerminalPublicationError) as caught:
        publication.build_publication_summary(document)

    assert caught.value.code == publication.FAIL_IDENTITY


@pytest.mark.parametrize(
    "field",
    (
        "role_evaluation_started",
        "enumerator_container_split_inputs_accessed",
        "raw_split_seed_accessed",
        "split_assignment_rows_accessed",
        "enumerator_container_target_inputs_accessed",
        "contains_private_key",
        "contains_raw_split_seed",
    ),
)
def test_forbidden_access_tamper_fails_closed(field: str) -> None:
    document = deepcopy(_load())
    document[field] = True
    _refresh_self_hash(document)

    with pytest.raises(publication.M3TerminalPublicationError) as caught:
        publication.build_publication_summary(document)

    assert caught.value.code == publication.FAIL_ACCESS


def test_role_evaluation_admission_tamper_fails_closed() -> None:
    document = deepcopy(_load())
    document["local_admission_artifact"]["role_evaluation_allowed"] = True
    _refresh_self_hash(document)

    with pytest.raises(publication.M3TerminalPublicationError) as caught:
        publication.build_publication_summary(document)

    assert caught.value.code == publication.FAIL_ACCESS


def test_formal_object_root_tamper_fails_closed() -> None:
    document = deepcopy(_load())
    document["formal_objects"]["dual_replay_agreement"]["content_root_hex"] = "0" * 64
    _refresh_self_hash(document)

    with pytest.raises(publication.M3TerminalPublicationError) as caught:
        publication.build_publication_summary(document)

    assert caught.value.code == publication.FAIL_FORMAL_ROOT


def test_python_rust_core_archive_mismatch_fails_closed() -> None:
    document = deepcopy(_load())
    rust_rows = document["archive_files"]["rust"]
    target = next(
        row
        for row in rust_rows
        if row["relative_path"]
        == "archive/canonical_program_records.cborframed"
    )
    target["sha256"] = "0" * 64
    _refresh_self_hash(document)

    with pytest.raises(publication.M3TerminalPublicationError) as caught:
        publication.build_publication_summary(document)

    assert caught.value.code == publication.FAIL_ARCHIVE


def test_unchecked_nested_tamper_still_fails_exact_mapping_hash() -> None:
    document = deepcopy(_load())
    document["runner_evidence_files"][0]["byte_length"] += 1
    _refresh_self_hash(document)

    with pytest.raises(publication.M3TerminalPublicationError) as caught:
        publication.build_publication_summary(document)

    assert caught.value.code == publication.FAIL_FILE_HASH
