from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
ARTIFACT = (
    PROJECT_ROOT
    / "artifacts/phase3_shrink2_dual_complete_enumeration_diagnostic_v1.json"
)
EVIDENCE_DOMAIN = (
    b"HEGEL/SHRINK2/DUAL_COMPLETE_ENUMERATION_DIAGNOSTIC/EVIDENCE/V1"
)
SOURCE_SET_DOMAIN = (
    b"HEGEL/SHRINK2/DUAL_COMPLETE_DIAGNOSTIC/COMMIT_H_FILE_SET/V1"
)
EXTERNAL_SET_DOMAIN = (
    b"HEGEL/SHRINK2/DUAL_COMPLETE_DIAGNOSTIC/EXTERNAL_ARTIFACT_SET/V1"
)


def _artifact() -> dict[str, object]:
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _git(*arguments: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return completed.stdout if binary else completed.stdout.decode().strip()


def _diagnostic_id(value: dict[str, object]) -> str:
    material = dict(value)
    material["evidence_record_id"] = None
    payload = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = sha256(EVIDENCE_DOMAIN + b"\x00" + payload).hexdigest()
    return f"phase3_shrink2_dual_complete_enumeration_diagnostic_{digest}"


def _source_set_root(
    source_files: list[dict[str, object]], commit_h: str
) -> str:
    payload = bytearray(SOURCE_SET_DOMAIN + b"\x00")
    paths = [str(item["path"]) for item in source_files]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    for item in source_files:
        path = str(item["path"])
        repository_path = f"Hegel Machine/{path}"
        content = _git("show", f"{commit_h}:{repository_path}", binary=True)
        assert isinstance(content, bytes)
        digest = sha256(content).hexdigest()
        assert item["sha256"] == f"sha256:{digest}"
        assert item["size"] == len(content)
        encoded = path.encode("utf-8")
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
        payload.extend(bytes.fromhex(digest))
    return f"sha256:{sha256(payload).hexdigest()}"


def _external_set_root(
    files: list[dict[str, object]], directory: Path
) -> str:
    payload = bytearray(EXTERNAL_SET_DOMAIN + b"\x00")
    paths = [str(item["path"]) for item in files]
    assert paths == sorted(paths, key=lambda path: PurePosixPath(path).parts)
    assert len(paths) == len(set(paths))
    for item in files:
        path = str(item["path"])
        content = (directory / path).read_bytes()
        digest = sha256(content).hexdigest()
        assert item["sha256"] == f"sha256:{digest}"
        assert item["size"] == len(content)
        encoded = path.encode("utf-8")
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
        payload.extend(len(content).to_bytes(8, "big"))
        payload.extend(bytes.fromhex(digest))
    return f"sha256:{sha256(payload).hexdigest()}"


def test_evidence_record_is_self_bound_and_commit_h_exact() -> None:
    value = _artifact()
    repository = value["repository_binding"]
    assert isinstance(repository, dict)
    commit_h = str(repository["commit_h"])

    assert value["evidence_record_id"] == _diagnostic_id(value)
    assert _git("rev-parse", f"{commit_h}^{{tree}}") == repository["commit_h_tree"]
    assert _git("show", "-s", "--format=%s", commit_h) == repository["commit_h_subject"]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit_h, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    source_files = repository["source_files"]
    assert isinstance(source_files, list)
    assert repository["source_file_set_root"] == _source_set_root(
        source_files, commit_h
    )
    assert repository["publication_remote_push_performed"] is False


def test_external_archive_manifest_replays_when_local_archive_is_present() -> None:
    value = _artifact()
    manifest = value["external_artifact_manifest"]
    assert isinstance(manifest, dict)
    files = manifest["files"]
    assert isinstance(files, list)
    directory = Path(str(manifest["local_archive_path"]))
    if not directory.is_dir():
        pytest.skip("commit-bound full streams are held in the local external archive")
    assert manifest["artifact_set_root"] == _external_set_root(files, directory)

    host = json.loads((directory / "host.stdout.json").read_text(encoding="utf-8"))
    result = value["dual_result"]
    replay = value["host_replay"]
    assert isinstance(result, dict)
    assert isinstance(replay, dict)
    assert host["closure_status"] == result["closure_status"]
    assert host["canonical_program_count"] == result["canonical_program_count"]
    assert (
        host["raw_operator_application_count"]
        == result["raw_operator_application_count"]
    )
    assert (
        f"sha256:{host['canonical_program_archive_root']}"
        == result["canonical_program_archive_root"]
    )
    assert host["archive_prefix_exact"] == replay["archive_prefix_exact"]
    assert (
        host["typed_language_boundary_independently_derived"]
        == replay["typed_language_boundary_independently_derived"]
    )


def test_claim_boundary_remains_non_formal_and_routes_only_to_shrink3() -> None:
    value = _artifact()
    result = value["dual_result"]
    runtime = value["runtime_isolation"]
    attempts = value["execution_attempts"]
    assert isinstance(result, dict)
    assert isinstance(runtime, dict)
    assert isinstance(attempts, dict)

    assert value["status"] == "DUAL_DSL_TOO_LARGE_HOST_REPLAY_PASS"
    assert value["claim_level"] == "NON_FORMAL_DUAL_CHILD_DIAGNOSTIC"
    assert value["qualification_level"] == "DIAGNOSTIC_ONLY"
    assert value["execution_state"] == "NOT_RUN"
    assert value["formal_roots"] is None
    assert value["formal_archive_roots_generated"] is False
    assert value["formal_closure_execution_performed"] is False
    assert value["formal_state_transition_allowed"] is False
    assert value["complete_closure_cardinality_established"] is False
    assert value["target_role_evaluation_performed"] is False
    assert value["outside_certificate_issued"] is False
    assert value["signature_bundle"] is None
    assert result["closure_status"] == "DSL_TOO_LARGE"
    assert result["canonical_program_count"] == 50_000
    assert result["first_out_of_budget_program_ordinal"] == 50_001
    assert result["raw_operator_application_count_scope"] == (
        "THROUGH_FULLY_CLOSED_BOUNDARY_BUCKET"
    )
    replay = value["host_replay"]
    assert isinstance(replay, dict)
    assert replay["independence_scope"] == (
        "INDEPENDENT_OF_ENDPOINT_REPORTED_WITNESS_NOT_A_THIRD_IMPLEMENTATION"
    )
    assert replay["residual_out_of_budget_canonical_programs_scope"] == (
        "CLOSED_BOUNDARY_BUCKET_ONLY"
    )
    assert runtime["container_network"] == "none"
    assert runtime["container_pull_policy"] == "never"
    assert runtime["network_access_performed"] is False
    assert runtime["host_target_or_split_modules_loaded"] is False
    assert attempts["host_replay"]["exit_code"] == 0
    assert attempts["python"]["exit_code"] == 0
    assert attempts["rust_attempts"][0]["enumeration_artifacts_published"] is False
    assert attempts["rust_attempts"][1]["exit_code"] == 0
    assert value["required_next_action"] == (
        "BEGIN_PREREGISTERED_SHRINK3_ENGINEERING_REMOVE_ADD_RETAIN_DIFFERENCE_"
        "WITHOUT_FORMAL_STATUS_PROMOTION"
    )
