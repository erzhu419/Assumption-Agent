from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tarfile

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hegel_machine.strict_ast_shrink5_v1 import (  # noqa: E402
    decode_shrink5_canonical_ast,
)


ARTIFACT = (
    PROJECT_ROOT
    / "artifacts/phase3_shrink5_dual_complete_enumeration_diagnostic_v1.json"
)
EVIDENCE_DOMAIN = (
    b"HEGEL/SHRINK5/DUAL_COMPLETE_ENUMERATION_DIAGNOSTIC/EVIDENCE/V1"
)
SOURCE_SET_DOMAIN = b"HEGEL/SHRINK5/DUAL_COMPLETE_SOURCE_SET/V1"
EXTERNAL_SET_DOMAIN = (
    b"HEGEL/SHRINK5/DUAL_COMPLETE_DIAGNOSTIC/EXTERNAL_ARTIFACT_SET/V1"
)
SUPERVISOR_DOMAIN = b"HEGEL/SHRINK5/DUAL_COMPLETE_SUPERVISOR/V1"
CARGO_SEED_DOMAIN = b"HEGEL/SHRINK5/CARGO_SEED_MANIFEST/V1"
SOURCE_U = "a3c384b4cb0f95583af6a1eb1c1d256ef6e9128a"
EVIDENCE_T = "01b66cd8effeab258797998f594b250188d823da"

ARTIFACT_FIELDS = {
    "active_governance_changed",
    "active_promotion_allowed",
    "artifact",
    "authoritative_claim_allowed",
    "claim_boundary",
    "claim_level",
    "complete_closure_cardinality_established",
    "diagnostic_evidence_not_signature",
    "dsl_version",
    "dual_result",
    "evidence_label",
    "evidence_record_id",
    "evidence_record_id_framing",
    "execution_state",
    "external_artifact_manifest",
    "formal_archive_roots_generated",
    "formal_closure_execution_performed",
    "formal_roots",
    "formal_roots_generated",
    "formal_state_transition_allowed",
    "freeze_version",
    "host_replay",
    "human_amendment_id",
    "keys_or_signatures_generated",
    "outside_certificate_issued",
    "qualification_level",
    "regression_tests",
    "repository_binding",
    "required_next_action",
    "routing",
    "runtime_isolation",
    "schema_version",
    "seed_material",
    "seeds_accessed",
    "shrink_step_id",
    "signature_bundle",
    "split_material_accessed",
    "status",
    "strict_qualification_binding",
    "supervisor_receipts",
    "target_role_evaluation_performed",
    "target_roles_evaluated",
}
EXTERNAL_PATHS = {
    "commit_bound_sources.tar",
    "diagnostic_outputs.tar",
    "host_replay_receipt.json",
    "python/result/bucket_accounting_records.cborframed",
    "python/result/canonical_program_records.cborframed",
    "python/result/program_chunk_manifests.cborframed",
    "python/result/report.json",
    "repository_binding.json",
    "rust/result/bucket_accounting_records.cborframed",
    "rust/result/canonical_program_records.cborframed",
    "rust/result/program_chunk_manifests.cborframed",
    "rust/result/report.json",
    "supervisor_summary.json",
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_json(payload: bytes) -> dict[str, object]:
    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    assert isinstance(value, dict)
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _validate_artifact_contract(value: dict[str, object]) -> None:
    assert set(value) == ARTIFACT_FIELDS
    result = value["dual_result"]
    regression = value["regression_tests"]
    assert isinstance(result, dict)
    assert isinstance(regression, dict)
    for field in (
        "active_governance_changed",
        "authoritative_claim_allowed",
        "formal_roots_generated",
        "formal_state_transition_allowed",
    ):
        assert type(value[field]) is bool
    for field in (
        "bucket_record_count",
        "canonical_program_count",
        "raw_operator_application_count",
    ):
        assert type(result[field]) is int
    assert type(regression["evidence_validation_passed"]) is int


def _artifact() -> dict[str, object]:
    payload = ARTIFACT.read_bytes()
    value = _strict_json(payload)
    assert payload == _canonical_json(value) + b"\n"
    _validate_artifact_contract(value)
    return value


def _evidence_record_id(value: dict[str, object]) -> str:
    material = dict(value)
    material["evidence_record_id"] = None
    digest = sha256(
        EVIDENCE_DOMAIN + b"\x00" + _canonical_json(material)
    ).hexdigest()
    return f"phase3_shrink5_dual_complete_enumeration_diagnostic_{digest}"


def _git(*arguments: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=not binary,
    )
    if binary:
        assert isinstance(completed.stdout, bytes)
        return completed.stdout
    assert isinstance(completed.stdout, str)
    return completed.stdout.strip()


def _external_directory(value: dict[str, object]) -> Path:
    manifest = value["external_artifact_manifest"]
    assert isinstance(manifest, dict)
    directory = Path(str(manifest["local_archive_path"]))
    if not directory.is_dir():
        pytest.skip("commit-bound full streams are held in the local external archive")
    return directory


def _external_set_root(
    files: list[dict[str, object]], directory: Path
) -> str:
    digest = sha256(EXTERNAL_SET_DOMAIN + b"\x00")
    paths = [str(row["path"]) for row in files]
    assert paths == sorted(paths, key=lambda path: PurePosixPath(path).parts)
    assert len(paths) == len(set(paths))
    for row in files:
        path = str(row["path"])
        target = directory / path
        metadata = target.lstat()
        assert stat.S_ISREG(metadata.st_mode)
        assert not target.is_symlink()
        payload = target.read_bytes()
        observed = sha256(payload).hexdigest()
        assert row == {
            "path": path,
            "sha256": f"sha256:{observed}",
            "size": len(payload),
        }
        encoded = path.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(bytes.fromhex(observed))
    return "sha256:" + digest.hexdigest()


def _source_set_root(rows: list[dict[str, object]]) -> str:
    digest = sha256(SOURCE_SET_DOMAIN + b"\x00")
    paths = [str(row["path"]) for row in rows]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    for row in rows:
        assert set(row) == {"path", "mode", "git_blob_oid", "sha256", "size"}
        fields = (
            str(row["path"]).encode("utf-8"),
            str(row["mode"]).encode("ascii"),
            bytes.fromhex(str(row["git_blob_oid"])),
            bytes.fromhex(str(row["sha256"])),
            int(row["size"]).to_bytes(8, "big"),
        )
        for field in fields:
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return "sha256:" + digest.hexdigest()


def _cargo_seed_manifest_root(rows: list[dict[str, object]]) -> str:
    digest = sha256(CARGO_SEED_DOMAIN + b"\x00")
    previous = ""
    for row in rows:
        assert set(row) == {"path", "mode", "size", "sha256"}
        path = str(row["path"])
        pure = PurePosixPath(path)
        assert (
            not pure.is_absolute()
            and pure.as_posix() == path
            and pure.parts[0] in {"cache", "index"}
            and path > previous
        )
        previous = path
        mode = str(row["mode"])
        size = int(row["size"])
        file_hash = str(row["sha256"])
        assert len(mode) == 4 and all(
            character in "01234567" for character in mode
        )
        assert 0 <= size < 1 << 64
        assert len(file_hash) == 64
        fields = (
            path.encode("utf-8"),
            mode.encode("ascii"),
            size.to_bytes(8, "big"),
            bytes.fromhex(file_hash),
        )
        for field in fields:
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return "sha256:" + digest.hexdigest()


def _frames(payload: bytes, expected_count: int) -> list[bytes]:
    records: list[bytes] = []
    offset = 0
    while offset < len(payload):
        assert len(payload) - offset >= 4
        length = int.from_bytes(payload[offset : offset + 4], "big")
        offset += 4
        assert 1 <= length <= 1_048_576
        assert offset + length <= len(payload)
        records.append(payload[offset : offset + length])
        offset += length
    assert offset == len(payload)
    assert len(records) == expected_count
    return records


def _rfc6962_root(records: list[bytes]) -> str:
    def subtree(first: int, last: int) -> bytes:
        count = last - first
        if count == 0:
            return sha256(b"").digest()
        if count == 1:
            return sha256(b"\x00" + records[first]).digest()
        split = 1 << ((count - 1).bit_length() - 1)
        return sha256(
            b"\x01"
            + subtree(first, first + split)
            + subtree(first + split, last)
        ).digest()

    return "sha256:" + subtree(0, len(records)).hex()


def test_evidence_v_is_self_bound_to_exact_source_u_and_has_no_formal_authority() -> None:
    value = _artifact()
    assert value["regression_tests"]["evidence_validation_passed"] == 4
    for payload in (b'{"x":1,"x":2}', b'{"x":NaN}'):
        with pytest.raises(ValueError):
            _strict_json(payload)
    for field, replacement in (
        ("formal_roots_generated", 0),
        ("active_governance_changed", 0),
    ):
        mutated = deepcopy(value)
        mutated[field] = replacement
        with pytest.raises(AssertionError):
            _validate_artifact_contract(mutated)
    mutated = deepcopy(value)
    mutated["dual_result"]["canonical_program_count"] = 50_000.0
    with pytest.raises(AssertionError):
        _validate_artifact_contract(mutated)
    mutated = deepcopy(value)
    mutated["dual_result"]["bucket_record_count"] = True
    with pytest.raises(AssertionError):
        _validate_artifact_contract(mutated)
    mutated = deepcopy(value)
    mutated["evidence_vecord_id"] = None
    with pytest.raises(AssertionError):
        _validate_artifact_contract(mutated)
    repository = value["repository_binding"]
    strict = value["strict_qualification_binding"]
    assert isinstance(repository, dict)
    assert isinstance(strict, dict)
    assert set(repository) == {
        "branch",
        "project_tree_oid",
        "publication_remote_push_performed",
        "repository_binding_file_sha256",
        "source_archive_sha256",
        "source_file_count",
        "source_file_set_framing",
        "source_file_set_root",
        "source_u_commit",
        "source_u_parent",
        "source_u_subject",
        "source_u_tree",
    }
    assert set(strict) == {
        "strict_qualification_artifact_path",
        "strict_qualification_artifact_sha256",
        "strict_qualification_diagnostic_report_hash",
        "strict_qualification_evidence_commit",
        "strict_qualification_source_commit",
        "strict_qualification_status",
    }
    assert value["evidence_record_id"] == _evidence_record_id(value)
    assert value["schema_version"] == (
        "hegel-phase3-shrink5-dual-complete-enumeration-diagnostic-evidence/1"
    )
    assert value["evidence_label"] == "Evidence V"
    assert repository["source_u_commit"] == SOURCE_U
    assert repository["source_u_parent"] == EVIDENCE_T
    assert _git("rev-parse", f"{SOURCE_U}^{{tree}}") == repository["source_u_tree"]
    assert _git("rev-parse", f"{SOURCE_U}:Hegel Machine") == repository[
        "project_tree_oid"
    ]
    assert _git("show", "-s", "--format=%P", SOURCE_U) == EVIDENCE_T
    assert _git("show", "-s", "--format=%s", SOURCE_U) == repository[
        "source_u_subject"
    ]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_U, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    assert repository["publication_remote_push_performed"] is False
    assert strict == {
        "strict_qualification_artifact_path": (
            "Hegel Machine/artifacts/phase3_m3_runtime/"
            "phase3_shrink5_sealed_dual_strict_qualification_v1.json"
        ),
        "strict_qualification_artifact_sha256": (
            "75761fc536d96d5d0bc91c5c0ba30dbc7c9ee21aac8d3f1dc5c96f6aca919b76"
        ),
        "strict_qualification_diagnostic_report_hash": (
            "sha256:5ee04b21477fd9f09271272fd6ecbf876b885b7831b37a868343a93996a187db"
        ),
        "strict_qualification_evidence_commit": EVIDENCE_T,
        "strict_qualification_source_commit": (
            "320b0a3458901090cb738023a4398220fb1d9277"
        ),
        "strict_qualification_status": "SEALED_DUAL_STRICT_OUTCOME_REPLAY_PASS",
    }
    assert value["status"] == "DUAL_DSL_TOO_LARGE_HOST_REPLAY_PASS"
    assert value["claim_level"] == "NON_FORMAL_DUAL_CHILD_DIAGNOSTIC"
    assert value["qualification_level"] == "DIAGNOSTIC_ONLY"
    assert value["execution_state"] == "NOT_RUN"
    assert value["formal_roots"] is None
    for field in (
        "formal_roots_generated",
        "formal_archive_roots_generated",
        "formal_closure_execution_performed",
        "formal_state_transition_allowed",
        "complete_closure_cardinality_established",
        "authoritative_claim_allowed",
        "active_promotion_allowed",
        "target_roles_evaluated",
        "target_role_evaluation_performed",
        "split_material_accessed",
        "seeds_accessed",
        "keys_or_signatures_generated",
        "outside_certificate_issued",
        "active_governance_changed",
    ):
        assert value[field] is False
    assert value["seed_material"] is None
    assert value["signature_bundle"] is None
    assert value["diagnostic_evidence_not_signature"] is True
    manifest = value["external_artifact_manifest"]
    result = value["dual_result"]
    assert isinstance(manifest, dict)
    assert isinstance(result, dict)
    files = manifest["files"]
    assert isinstance(files, list)
    assert manifest["file_count"] == len(files) == 13
    assert manifest["total_bytes"] == sum(int(row["size"]) for row in files)
    assert {str(row["path"]) for row in files} == EXTERNAL_PATHS
    assert all(set(row) == {"path", "sha256", "size"} for row in files)
    assert result["closure_status"] == "DSL_TOO_LARGE"
    assert result["canonical_program_count"] == 50_000
    assert result["raw_operator_application_count"] == 3_120_719
    assert result["first_out_of_budget_program_ordinal"] == 50_001
    assert result["residual_out_of_budget_canonical_programs"] == 2_237
    assert value["routing"]["operation"] == (
        "reduce max_total_ast_depth from 4 to 3"
    )
    assert value["routing"]["only_open_route"] is True
    assert ARTIFACT.stat().st_size < 20_000


def test_external_manifest_and_git_archive_replay_exact_source_u() -> None:
    value = _artifact()
    directory = _external_directory(value)
    manifest = value["external_artifact_manifest"]
    repository = value["repository_binding"]
    assert isinstance(manifest, dict)
    assert isinstance(repository, dict)
    assert set(manifest) == {
        "artifact_set_framing",
        "artifact_set_root",
        "file_count",
        "files",
        "local_archive_path",
        "total_bytes",
    }
    files = manifest["files"]
    assert isinstance(files, list)
    assert manifest["file_count"] == len(files) == 13
    assert {str(row["path"]) for row in files} == EXTERNAL_PATHS
    assert manifest["total_bytes"] == sum(int(row["size"]) for row in files)
    actual_paths = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    assert actual_paths == EXTERNAL_PATHS
    assert not any(path.is_symlink() for path in directory.rglob("*"))
    assert manifest["artifact_set_root"] == _external_set_root(files, directory)

    summary = _strict_json((directory / "supervisor_summary.json").read_bytes())
    binding_bytes = (directory / "repository_binding.json").read_bytes()
    binding = _strict_json(binding_bytes)
    assert len(summary) == 28
    assert len(binding) == 9
    assert binding == summary["repository_binding"]
    assert repository["repository_binding_file_sha256"] == (
        "sha256:" + sha256(binding_bytes).hexdigest()
    )
    assert binding["basis_commit"] == SOURCE_U
    assert binding["basis_parent_commits"] == [EVIDENCE_T]
    rows = binding["source_files"]
    assert isinstance(rows, list)
    assert len(rows) == binding["source_file_count"] == repository["source_file_count"]
    assert binding["source_file_set_root"] == repository["source_file_set_root"]
    assert binding["source_file_set_root"] == _source_set_root(rows)
    for row in rows:
        path = str(row["path"])
        listing = _git("ls-tree", SOURCE_U, "--", path)
        assert isinstance(listing, str)
        metadata, listed_path = listing.split("\t", 1)
        mode, kind, oid = metadata.split(" ")
        assert (listed_path, kind, mode, oid) == (
            path,
            "blob",
            row["mode"],
            row["git_blob_oid"],
        )
        content = _git("show", f"{SOURCE_U}:{path}", binary=True)
        assert isinstance(content, bytes)
        assert len(content) == row["size"]
        assert sha256(content).hexdigest() == row["sha256"]

    source_tar = (directory / "commit_bound_sources.tar").read_bytes()
    assert repository["source_archive_sha256"] == (
        "sha256:" + sha256(source_tar).hexdigest()
    )
    with tarfile.open(fileobj=io.BytesIO(source_tar), mode="r:") as archive:
        members = archive.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            assert not path.is_absolute()
            assert ".." not in path.parts
            assert member.isfile() or member.isdir()
        regular = {member.name: member for member in members if member.isfile()}
        assert set(regular) == {str(row["path"]) for row in rows}
        for row in rows:
            extracted = archive.extractfile(regular[str(row["path"])])
            assert extracted is not None
            content = extracted.read()
            assert len(content) == row["size"]
            assert sha256(content).hexdigest() == row["sha256"]


def test_supervisor_reports_and_offline_isolation_match_evidence_v() -> None:
    value = _artifact()
    directory = _external_directory(value)
    receipts = value["supervisor_receipts"]
    runtime = value["runtime_isolation"]
    result = value["dual_result"]
    assert isinstance(receipts, dict)
    assert isinstance(runtime, dict)
    assert isinstance(result, dict)
    assert set(receipts) == {
        "diagnostic_outputs_archive_sha256",
        "diagnostic_summary_hash",
        "host_replay_receipt_sha256",
        "python_endpoint_stdout_sha256",
        "python_report_sha256",
        "repository_binding_file_sha256",
        "rust_endpoint_stdout_sha256",
        "rust_report_sha256",
        "schema_version",
        "status",
        "supervisor_summary_sha256",
    }

    paths = {
        "supervisor_summary_sha256": "supervisor_summary.json",
        "host_replay_receipt_sha256": "host_replay_receipt.json",
        "repository_binding_file_sha256": "repository_binding.json",
        "python_report_sha256": "python/result/report.json",
        "rust_report_sha256": "rust/result/report.json",
        "diagnostic_outputs_archive_sha256": "diagnostic_outputs.tar",
    }
    for field, path in paths.items():
        payload = (directory / path).read_bytes()
        assert receipts[field] == "sha256:" + sha256(payload).hexdigest()
    summary = _strict_json((directory / "supervisor_summary.json").read_bytes())
    summary_material = dict(summary)
    summary_hash = summary_material.pop("diagnostic_summary_hash")
    assert summary_hash == (
        "sha256:"
        + sha256(
            SUPERVISOR_DOMAIN + b"\x00" + _canonical_json(summary_material)
        ).hexdigest()
    )
    assert summary_hash == receipts["diagnostic_summary_hash"]
    assert summary["schema_version"] == receipts["schema_version"]
    assert summary["status"] == receipts["status"]
    assert summary["basis_commit"] == SOURCE_U
    assert receipts["python_endpoint_stdout_sha256"] == (
        "sha256:" + summary["python_endpoint_stdout_sha256"]
    )
    assert receipts["rust_endpoint_stdout_sha256"] == (
        "sha256:" + summary["rust_endpoint_stdout_sha256"]
    )

    host = _strict_json((directory / "host_replay_receipt.json").read_bytes())
    python_payload = (directory / "python/result/report.json").read_bytes()
    rust_payload = (directory / "rust/result/report.json").read_bytes()
    python = _strict_json(python_payload)
    rust = _strict_json(rust_payload)
    assert len(host) == 57
    assert python_payload == _canonical_json(python) + b"\n"
    assert rust_payload == _canonical_json(rust) + b"\n"
    assert summary["host_replay_receipt"] == host
    assert host["python_report_sha256"] == receipts["python_report_sha256"][7:]
    assert host["rust_report_sha256"] == receipts["rust_report_sha256"][7:]
    assert receipts["python_endpoint_stdout_sha256"] == receipts[
        "python_report_sha256"
    ]
    assert receipts["rust_endpoint_stdout_sha256"] == receipts[
        "rust_report_sha256"
    ]
    assert len(python) == result["report_field_counts"]["python"] == 78
    assert len(rust) == result["report_field_counts"]["rust"] == 75
    python_common = dict(python)
    for field in (
        "loaded_hegel_modules",
        "target_free_isolation_verified",
        "target_or_split_modules_loaded",
    ):
        python_common.pop(field)
    for report in (python_common, rust):
        for field in (
            "schema_version",
            "implementation",
            "implementation_id",
            "implementation_machine_id",
        ):
            report.pop(field)
    assert python_common == rust

    for actor in (summary, host, python, rust):
        assert actor["execution_state"] == "NOT_RUN"
        assert actor["formal_roots"] is None
        assert actor["formal_roots_generated"] is False
        assert actor["authoritative_claim_allowed"] is False
        assert actor["target_roles_evaluated"] is False
        assert actor["split_material_accessed"] is False
    assert summary["formal_state_transition_allowed"] is False
    assert summary["seeds_accessed"] is False
    assert summary["keys_or_signatures_generated"] is False
    assert summary["active_governance_changed"] is False
    assert host["formal_state_transition_allowed"] is False
    assert host["secrets_accessed"] is False
    assert python["secrets_accessed"] is False
    assert rust["secrets_accessed"] is False

    container = summary["container_runtime"]
    cargo = container["cargo_dependency_seed"]
    assert container["network"] == runtime["container_network"] == "none"
    assert container["pull_policy"] == runtime["container_pull_policy"] == "never"
    assert container["python_image"] == runtime["python_container_image"]
    assert container["rust_image"] == runtime["rust_container_image"]
    assert summary["parallel_endpoint_count"] == runtime["parallel_endpoint_count"] == 2
    assert summary["parallel_endpoint_execution"] is True
    assert runtime["network_access_performed"] is False
    assert runtime["user_secrets_mounted"] is False
    assert runtime["organizational_independence"] is False
    assert runtime["same_admin_controller"] is True
    assert runtime["technical_role_independence"] is True
    assert cargo["file_count"] == runtime["cargo_dependency_seed_file_count"]
    assert cargo["manifest_root"] == runtime["cargo_dependency_seed_manifest_root"]
    assert cargo["manifest_root"] == _cargo_seed_manifest_root(cargo["files"])
    assert cargo["file_count"] == len(cargo["files"]) == 43
    assert cargo["total_byte_count"] == sum(
        int(row["size"]) for row in cargo["files"]
    )
    assert cargo["pre_build_manifest_root"] == cargo["post_build_manifest_root"]
    assert cargo["pre_build_manifest_root"] == runtime[
        "cargo_dependency_pre_build_manifest_root"
    ]
    assert cargo["post_build_manifest_root"] == runtime[
        "cargo_dependency_post_build_manifest_root"
    ]
    assert cargo["manifest_unchanged_after_build"] is True
    assert cargo["permission_frozen_reverified"] is True
    assert all(row["mode"] == "0444" for row in cargo["files"])
    assert cargo["selected_subtrees"] == ["cache", "index"]
    assert cargo["src_subtree_included"] is False
    assert cargo["verified_crate_count"] == 21
    assert cargo["fresh_tmpfs_cargo_home"] is True
    assert container["target_volume"]["fresh_before_run"] is True
    assert container["target_volume_removed_after_run"] is True

    actor_profile = _strict_json(
        _git(
            "show",
            f"{SOURCE_U}:Hegel Machine/config/phase3_container_actor_profile_v1.json",
            binary=True,
        )
    )
    build_profile = _strict_json(
        _git(
            "show",
            f"{SOURCE_U}:Hegel Machine/config/phase3_shrink5_enumerator_offline_build_profile_v1.json",
            binary=True,
        )
    )
    assert actor_profile["profile_id"] == runtime["actor_profile_id"]
    for flag in (
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=65534:65534",
    ):
        assert flag in actor_profile["required_runtime_flags"]
    assert build_profile["profile_id"] == runtime["enumerator_build_profile_id"]
    assert build_profile["network"] == "none"
    assert build_profile["pull_policy"] == "never"
    assert build_profile["cargo_flags"] == ["--release", "--locked", "--offline"]


def test_framed_stream_roots_boundary_and_only_step6_route_replay() -> None:
    value = _artifact()
    directory = _external_directory(value)
    result = value["dual_result"]
    replay = value["host_replay"]
    host = _strict_json((directory / "host_replay_receipt.json").read_bytes())
    assert isinstance(result, dict)
    assert isinstance(replay, dict)
    assert set(replay) == {
        "archive_prefix_exact",
        "automatic_operator_migration_performed",
        "binary_operator_registry_verified",
        "bucket_accounting_verified",
        "chunk_framing_and_blob_hashes_verified",
        "dual_archive_bytes_equal",
        "dual_reports_equal_after_endpoint_normalization",
        "host_strict_archive_replay_verified",
        "independence_scope",
        "operator_id_compaction_performed",
        "post_witness_traversal_buckets_untouched",
        "program_binding_roots_verified",
        "program_indices_verified",
        "removed_binary_operator_absent_from_archive",
        "typed_language_boundary_independently_derived",
        "witness_adjacency_verified",
        "witness_closed_bucket_rank_verified",
    }

    stream_specs = {
        "canonical_program_records": (
            "canonical_program_records.cborframed",
            50_000,
            "canonical_program_archive_root",
        ),
        "program_chunk_manifests": (
            "program_chunk_manifests.cborframed",
            13,
            "program_chunk_manifest_root",
        ),
        "bucket_accounting_records": (
            "bucket_accounting_records.cborframed",
            150,
            "bucket_accounting_root",
        ),
    }
    for name, (filename, count, root_field) in stream_specs.items():
        python_payload = (directory / "python/result" / filename).read_bytes()
        rust_payload = (directory / "rust/result" / filename).read_bytes()
        assert python_payload == rust_payload
        assert result["stream_sha256"][name] == (
            "sha256:" + sha256(python_payload).hexdigest()
        )
        frames = _frames(python_payload, count)
        assert _rfc6962_root(frames) == result[root_field]

    output_tar = (directory / "diagnostic_outputs.tar").read_bytes()
    expected_tar_paths = {
        f"{role}/result/{filename}"
        for role in ("python", "rust")
        for filename in (
            "bucket_accounting_records.cborframed",
            "canonical_program_records.cborframed",
            "program_chunk_manifests.cborframed",
            "report.json",
        )
    } | {"host_replay_receipt.json"}
    with tarfile.open(fileobj=io.BytesIO(output_tar), mode="r:") as archive:
        members = archive.getmembers()
        assert {member.name for member in members} == expected_tar_paths
        assert all(member.isfile() for member in members)
        for member in members:
            assert member.uid == member.gid == 0
            assert member.mode == 0o444
            assert member.mtime == 0
            extracted = archive.extractfile(member)
            assert extracted is not None
            assert extracted.read() == (directory / member.name).read_bytes()

    assert result["closure_status"] == host["closure_status"] == "DSL_TOO_LARGE"
    assert result["canonical_program_count"] == host["canonical_program_count"] == 50_000
    assert result["raw_operator_application_count"] == 3_120_719
    assert result["raw_operator_application_count_scope"] == (
        "THROUGH_FULLY_CLOSED_BOUNDARY_BUCKET"
    )
    assert result["first_out_of_budget_program_ordinal"] == 50_001
    assert result["residual_out_of_budget_canonical_programs"] == 2_237
    assert result["residual_out_of_budget_canonical_programs_scope"] == (
        "CLOSED_BOUNDARY_BUCKET_ONLY"
    )
    witness = decode_shrink5_canonical_ast(
        bytes.fromhex(str(result["first_out_of_budget_program_cbor_hex"]))
    )
    assert result["first_out_of_budget_program_hash"] == (
        "sha256:" + witness.digest.hex()
    )
    assert result["maximum_top_level_clauses"] == 2
    assert result["and3_generator_attempts_allowed"] is False
    assert result["and3_raw_operator_application_count"] == 0
    assert result["maximum_ast_depth"] == 4
    assert result["maximum_ast_node_count"] == 6
    for field, expected in replay.items():
        host_field = (
            "dual_reports_equal"
            if field == "dual_reports_equal_after_endpoint_normalization"
            else field
        )
        assert host[host_field] == expected

    assert value["routing"] == {
        "authority": "ENGINEERING_ONLY",
        "formal_status_promotion_allowed": False,
        "from_max_total_ast_depth": 4,
        "maximum_ast_node_count_remains": 6,
        "maximum_top_level_clauses_remains": 2,
        "only_open_route": True,
        "operation": "reduce max_total_ast_depth from 4 to 3",
        "preregistered_shrink_order_step": 6,
        "to_max_total_ast_depth": 3,
    }
    assert value["required_next_action"] == (
        "BEGIN_PREREGISTERED_SHRINK_STEP_6_ENGINEERING_REDUCE_MAX_TOTAL_AST_"
        "DEPTH_FROM_4_TO_3_WITHOUT_FORMAL_STATUS_PROMOTION"
    )
