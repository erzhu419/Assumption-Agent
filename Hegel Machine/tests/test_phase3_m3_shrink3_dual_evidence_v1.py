from __future__ import annotations

from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tarfile

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
ARTIFACT = (
    PROJECT_ROOT
    / "artifacts/phase3_shrink3_dual_complete_enumeration_diagnostic_v1.json"
)
EVIDENCE_DOMAIN = (
    b"HEGEL/SHRINK3/DUAL_COMPLETE_ENUMERATION_DIAGNOSTIC/EVIDENCE/V1"
)
SOURCE_SET_DOMAIN = b"HEGEL/SHRINK3/DUAL_COMPLETE_SOURCE_SET/V1"
EXTERNAL_SET_DOMAIN = (
    b"HEGEL/SHRINK3/DUAL_COMPLETE_DIAGNOSTIC/EXTERNAL_ARTIFACT_SET/V1"
)
SUPERVISOR_DOMAIN = b"HEGEL/SHRINK3/DUAL_COMPLETE_SUPERVISOR/V1"


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


def _artifact() -> dict[str, object]:
    return _strict_json(ARTIFACT.read_bytes())


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_bytes(*arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _evidence_record_id(value: dict[str, object]) -> str:
    material = dict(value)
    material["evidence_record_id"] = None
    digest = sha256(
        EVIDENCE_DOMAIN + b"\x00" + _canonical_json(material)
    ).hexdigest()
    return f"phase3_shrink3_dual_complete_enumeration_diagnostic_{digest}"


def _external_set_root(
    files: list[dict[str, object]], directory: Path
) -> str:
    payload = bytearray(EXTERNAL_SET_DOMAIN + b"\x00")
    paths = [str(item["path"]) for item in files]
    assert paths == sorted(paths, key=lambda path: PurePosixPath(path).parts)
    assert len(paths) == len(set(paths))
    for item in files:
        path = str(item["path"])
        target = directory / path
        mode = target.lstat().st_mode
        assert stat.S_ISREG(mode)
        assert not target.is_symlink()
        content = target.read_bytes()
        digest = sha256(content).hexdigest()
        assert item["sha256"] == f"sha256:{digest}"
        assert item["size"] == len(content)
        encoded = path.encode("utf-8")
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
        payload.extend(len(content).to_bytes(8, "big"))
        payload.extend(bytes.fromhex(digest))
    return f"sha256:{sha256(payload).hexdigest()}"


def _source_set_root(rows: list[dict[str, object]]) -> str:
    digest = sha256(SOURCE_SET_DOMAIN + b"\x00")
    paths = [str(row["path"]) for row in rows]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    for row in rows:
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
    return f"sha256:{digest.hexdigest()}"


def _external_directory(value: dict[str, object]) -> Path:
    manifest = value["external_artifact_manifest"]
    assert isinstance(manifest, dict)
    directory = Path(str(manifest["local_archive_path"]))
    if not directory.is_dir():
        pytest.skip("commit-bound full streams are held in the local external archive")
    return directory


def test_evidence_record_is_self_bound_and_source_commit_m_is_exact() -> None:
    value = _artifact()
    repository = value["repository_binding"]
    assert isinstance(repository, dict)
    commit_m = str(repository["commit_m"])

    assert value["evidence_record_id"] == _evidence_record_id(value)
    assert _git("rev-parse", f"{commit_m}^{{tree}}") == repository["commit_m_tree"]
    assert _git("rev-parse", f"{commit_m}:Hegel Machine") == repository["project_tree_oid"]
    assert _git("show", "-s", "--format=%P", commit_m) == repository["commit_m_parent"]
    assert _git("show", "-s", "--format=%s", commit_m) == repository["commit_m_subject"]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit_m, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    assert repository["publication_remote_push_performed"] is False


def test_external_archive_manifest_source_tar_and_supervisor_replay() -> None:
    value = _artifact()
    directory = _external_directory(value)
    manifest = value["external_artifact_manifest"]
    repository = value["repository_binding"]
    receipts = value["supervisor_receipts"]
    assert isinstance(manifest, dict)
    assert isinstance(repository, dict)
    assert isinstance(receipts, dict)
    files = manifest["files"]
    assert isinstance(files, list)
    assert manifest["file_count"] == len(files) == 13
    assert manifest["total_bytes"] == sum(int(row["size"]) for row in files)
    actual_paths: set[str] = set()
    for path in directory.rglob("*"):
        assert not path.is_symlink()
        if path.is_file():
            assert stat.S_ISREG(path.lstat().st_mode)
            actual_paths.add(path.relative_to(directory).as_posix())
    assert actual_paths == {str(row["path"]) for row in files}
    assert manifest["artifact_set_root"] == _external_set_root(files, directory)

    summary_bytes = (directory / "supervisor_summary.json").read_bytes()
    assert receipts["supervisor_summary_sha256"] == (
        f"sha256:{sha256(summary_bytes).hexdigest()}"
    )
    summary = _strict_json(summary_bytes)
    summary_material = dict(summary)
    summary_hash = summary_material.pop("diagnostic_summary_hash")
    assert summary_hash == (
        "sha256:"
        + sha256(
            SUPERVISOR_DOMAIN + b"\x00" + _canonical_json(summary_material)
        ).hexdigest()
    )
    assert summary_hash == receipts["diagnostic_summary_hash"]
    assert summary["status"] == receipts["status"]
    assert summary["basis_commit"] == repository["commit_m"]
    assert summary["repository_binding"]["source_file_set_root"] == (
        repository["source_file_set_root"]
    )

    binding_bytes = (directory / "repository_binding.json").read_bytes()
    assert repository["repository_binding_file_sha256"] == (
        f"sha256:{sha256(binding_bytes).hexdigest()}"
    )
    binding = _strict_json(binding_bytes)
    assert binding == summary["repository_binding"]
    rows = binding["source_files"]
    assert binding["source_file_count"] == len(rows) == repository["source_file_count"]
    assert binding["source_file_set_root"] == _source_set_root(rows)
    commit_m = str(repository["commit_m"])
    for row in rows:
        path = str(row["path"])
        listing = _git("ls-tree", commit_m, "--", path)
        metadata, listed_path = listing.split("\t", 1)
        mode, kind, oid = metadata.split(" ")
        assert listed_path == path
        assert kind == "blob"
        assert mode == row["mode"]
        assert oid == row["git_blob_oid"]
        content = _git_bytes("show", f"{commit_m}:{path}")
        assert len(content) == row["size"]
        assert sha256(content).hexdigest() == row["sha256"]

    source_tar = (directory / "commit_bound_sources.tar").read_bytes()
    assert repository["source_archive_sha256"] == (
        f"sha256:{sha256(source_tar).hexdigest()}"
    )
    assert binding["source_archive_sha256"] == sha256(source_tar).hexdigest()
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


def test_dual_archives_and_host_replay_match_the_committed_result() -> None:
    value = _artifact()
    directory = _external_directory(value)
    result = value["dual_result"]
    replay = value["host_replay"]
    receipts = value["supervisor_receipts"]
    assert isinstance(result, dict)
    assert isinstance(replay, dict)
    assert isinstance(receipts, dict)
    assert result["maximum_canonical_programs"] == 50_000
    assert result["maximum_raw_operator_applications"] == 5_000_000

    host_bytes = (directory / "host_replay_receipt.json").read_bytes()
    python_bytes = (directory / "python/result/report.json").read_bytes()
    rust_bytes = (directory / "rust/result/report.json").read_bytes()
    assert receipts["host_replay_receipt_sha256"] == (
        f"sha256:{sha256(host_bytes).hexdigest()}"
    )
    assert receipts["python_report_sha256"] == (
        f"sha256:{sha256(python_bytes).hexdigest()}"
    )
    assert receipts["rust_report_sha256"] == (
        f"sha256:{sha256(rust_bytes).hexdigest()}"
    )
    host = _strict_json(host_bytes)
    python = _strict_json(python_bytes)
    rust = _strict_json(rust_bytes)
    summary = _strict_json((directory / "supervisor_summary.json").read_bytes())
    assert summary["host_replay_receipt"] == host
    assert host["python_report_sha256"] == sha256(python_bytes).hexdigest()
    assert host["rust_report_sha256"] == sha256(rust_bytes).hexdigest()
    assert receipts["python_endpoint_stdout_sha256"] == (
        f"sha256:{summary['python_endpoint_stdout_sha256']}"
    )
    assert receipts["rust_endpoint_stdout_sha256"] == (
        f"sha256:{summary['rust_endpoint_stdout_sha256']}"
    )
    assert summary["execution_state"] == "NOT_RUN"
    assert summary["formal_roots"] is None
    assert summary["formal_roots_generated"] is False
    assert summary["formal_state_transition_allowed"] is False
    assert summary["authoritative_claim_allowed"] is False
    assert summary["target_roles_evaluated"] is False
    assert summary["split_material_accessed"] is False
    assert summary["seeds_accessed"] is False
    assert summary["keys_or_signatures_generated"] is False
    assert summary["active_governance_changed"] is False
    assert host["execution_state"] == "NOT_RUN"
    assert host["formal_roots"] is None
    assert host["formal_roots_generated"] is False
    assert host["formal_state_transition_allowed"] is False
    assert host["authoritative_claim_allowed"] is False
    assert host["target_roles_evaluated"] is False
    assert host["split_material_accessed"] is False
    assert host["secrets_accessed"] is False

    runtime = value["runtime_isolation"]
    assert isinstance(runtime, dict)
    container = summary["container_runtime"]
    assert isinstance(container, dict)
    actor_profiles = container["actor_profiles"]
    build_profile = container["enumerator_build_profile"]
    cargo_seed = container["cargo_dependency_seed"]
    target_volume = container["target_volume"]
    assert isinstance(actor_profiles, dict)
    assert isinstance(build_profile, dict)
    assert isinstance(cargo_seed, dict)
    assert isinstance(target_volume, dict)
    assert container["python_image"] == runtime["python_container_image"]
    assert container["rust_image"] == runtime["rust_container_image"]
    assert container["network"] == runtime["container_network"]
    assert container["pull_policy"] == runtime["container_pull_policy"]
    assert actor_profiles["actor_profile_id"] == runtime["actor_profile_id"]
    assert build_profile["profile_id"] == runtime["enumerator_build_profile_id"]
    assert cargo_seed["file_count"] == runtime["cargo_dependency_seed_file_count"]
    assert cargo_seed["file_set_root"] == (
        runtime["cargo_dependency_seed_file_set_root"]
    )
    assert cargo_seed["selected_subtrees"] == (
        runtime["cargo_dependency_seed_selected_subtrees"]
    )
    assert cargo_seed["snapshot_reverified_after_copy"] == (
        runtime["cargo_dependency_seed_reverified_after_copy"]
    )
    assert cargo_seed["src_subtree_included"] == (
        runtime["cargo_dependency_src_subtree_mounted"]
    )
    assert cargo_seed["fresh_tmpfs_cargo_home"] == (
        runtime["fresh_tmpfs_cargo_home"]
    )
    assert cargo_seed["verified_crate_count"] == (
        runtime["cargo_locked_registry_packages_verified"]
    )
    assert summary["rust_release_binary_sha256"] == (
        str(runtime["rust_binary_sha256"])[7:]
    )
    assert summary["parallel_endpoint_count"] == runtime["parallel_endpoint_count"]
    assert summary["parallel_endpoint_execution"] == (
        runtime["parallel_endpoint_execution"]
    )
    assert target_volume["name"] == runtime["target_volume_name"]
    assert target_volume["fresh_before_run"] == (
        runtime["target_volume_fresh_before_run"]
    )
    assert container["target_volume_removed_after_run"] == (
        runtime["target_volume_removed_after_run"]
    )
    assert container["daemon_receipt"]["server"]["Version"] == (
        runtime["docker_server_version"]
    )

    repository = value["repository_binding"]
    assert isinstance(repository, dict)
    commit_m = str(repository["commit_m"])
    actor_profile = _strict_json(
        _git_bytes(
            "show",
            f"{commit_m}:Hegel Machine/config/phase3_container_actor_profile_v1.json",
        )
    )
    enumerator_profile = _strict_json(
        _git_bytes(
            "show",
            f"{commit_m}:Hegel Machine/config/phase3_shrink3_enumerator_offline_build_profile_v1.json",
        )
    )
    assert actor_profile["profile_id"] == runtime["actor_profile_id"]
    required_flags = actor_profile["required_runtime_flags"]
    assert isinstance(required_flags, list)
    for flag in (
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--user=65534:65534",
    ):
        assert flag in required_flags
    assert enumerator_profile["profile_id"] == (
        runtime["enumerator_build_profile_id"]
    )
    assert enumerator_profile["cargo_flags"] == runtime["rust_cargo_flags"]
    assert enumerator_profile["root_filesystem_read_only"] is True
    assert enumerator_profile["capabilities_dropped"] == "ALL"
    assert enumerator_profile["no_new_privileges"] is True

    assert len(python) == result["report_field_counts"]["python"] == 69
    assert len(rust) == result["report_field_counts"]["rust"] == 66
    for report in (python, rust):
        assert report["closure_status"] == result["closure_status"]
        assert report["canonical_program_count"] == result["canonical_program_count"]
        assert report["raw_operator_application_count"] == (
            result["raw_operator_application_count"]
        )
        assert report["maximum_canonical_programs"] == (
            result["maximum_canonical_programs"]
        )
        assert report["maximum_raw_operator_applications"] == (
            result["maximum_raw_operator_applications"]
        )
        assert report["first_out_of_budget_program_ordinal_or_null"] == 50_001
        assert report["first_out_of_budget_program_hash_or_null"] == (
            str(result["first_out_of_budget_program_hash"])[7:]
        )
        assert report["first_out_of_budget_program_cbor_hex_or_null"] == (
            result["first_out_of_budget_program_cbor_hex"]
        )
        assert report["closure_cardinality_or_null"] is None
        assert report["frontier_exhausted"] is False
        assert report["all_type_buckets_closed"] is False
        assert report["raw_expansion_limit_hit"] is False
        assert report["wall_clock_abort_hit"] is False
        assert report["traversal_prefix_complete"] is True
        assert report["formal_roots"] is None
        assert report["formal_roots_generated"] is False
        assert report["execution_state"] == "NOT_RUN"
        assert report["authoritative_claim_allowed"] is False
        assert report["target_roles_evaluated"] is False
        assert report["split_material_accessed"] is False
        assert report["secrets_accessed"] is False

    stream_names = {
        "bucket_accounting_records": "bucket_accounting_records.cborframed",
        "canonical_program_records": "canonical_program_records.cborframed",
        "program_chunk_manifests": "program_chunk_manifests.cborframed",
    }
    for key, filename in stream_names.items():
        python_stream = (directory / "python/result" / filename).read_bytes()
        rust_stream = (directory / "rust/result" / filename).read_bytes()
        assert python_stream == rust_stream
        assert result["stream_sha256"][key] == (
            f"sha256:{sha256(python_stream).hexdigest()}"
        )

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
    output_tar = (directory / "diagnostic_outputs.tar").read_bytes()
    output_tar_hash = sha256(output_tar).hexdigest()
    assert receipts["diagnostic_outputs_archive_sha256"] == (
        f"sha256:{output_tar_hash}"
    )
    assert summary["retained_local_files"][
        "diagnostic_outputs_archive_sha256"
    ] == output_tar_hash
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

    assert host["closure_status"] == result["closure_status"]
    assert host["canonical_program_count"] == result["canonical_program_count"]
    assert host["raw_operator_application_count"] == (
        result["raw_operator_application_count"]
    )
    assert host["raw_operator_application_count_scope"] == (
        result["raw_operator_application_count_scope"]
    )
    assert host["first_out_of_budget_program_hash"] == (
        str(result["first_out_of_budget_program_hash"])[7:]
    )
    assert host["first_out_of_budget_program_cbor_hex"] == (
        result["first_out_of_budget_program_cbor_hex"]
    )
    assert f"sha256:{host['canonical_program_archive_root']}" == (
        result["canonical_program_archive_root"]
    )
    assert f"sha256:{host['program_chunk_manifest_root']}" == (
        result["program_chunk_manifest_root"]
    )
    assert f"sha256:{host['bucket_accounting_root']}" == (
        result["bucket_accounting_root"]
    )
    assert host["archive_prefix_exact"] == replay["archive_prefix_exact"]
    assert host["typed_language_boundary_independently_derived"] == (
        replay["typed_language_boundary_independently_derived"]
    )
    for key, expected in replay.items():
        host_key = (
            "dual_reports_equal"
            if key == "dual_reports_equal_after_endpoint_normalization"
            else key
        )
        assert host[host_key] == expected
    assert host["removed_binary_operator_absent_from_archive"] is True
    assert host["residual_out_of_budget_canonical_programs"] == (
        result["residual_out_of_budget_canonical_programs"]
    )


def test_claim_boundary_remains_non_formal_and_routes_only_to_shrink4() -> None:
    value = _artifact()
    result = value["dual_result"]
    runtime = value["runtime_isolation"]
    replay = value["host_replay"]
    assert isinstance(result, dict)
    assert isinstance(runtime, dict)
    assert isinstance(replay, dict)

    assert value["status"] == "DUAL_DSL_TOO_LARGE_HOST_REPLAY_PASS"
    assert value["claim_level"] == "NON_FORMAL_DUAL_CHILD_DIAGNOSTIC"
    assert value["qualification_level"] == "DIAGNOSTIC_ONLY"
    assert value["execution_state"] == "NOT_RUN"
    assert value["formal_roots"] is None
    assert value["formal_archive_roots_generated"] is False
    assert value["formal_roots_generated"] is False
    assert value["formal_closure_execution_performed"] is False
    assert value["formal_state_transition_allowed"] is False
    assert value["complete_closure_cardinality_established"] is False
    assert value["target_role_evaluation_performed"] is False
    assert value["outside_certificate_issued"] is False
    assert value["signature_bundle"] is None
    assert value["seed_material"] is None
    assert value["authoritative_claim_allowed"] is False
    assert value["active_governance_changed"] is False
    assert value["keys_or_signatures_generated"] is False
    assert value["split_material_accessed"] is False
    assert value["seeds_accessed"] is False
    assert result["closure_status"] == "DSL_TOO_LARGE"
    assert result["canonical_program_count"] == 50_000
    assert result["first_out_of_budget_program_ordinal"] == 50_001
    assert result["raw_operator_application_count_scope"] == (
        "THROUGH_FULLY_CLOSED_BOUNDARY_BUCKET"
    )
    assert result["residual_out_of_budget_canonical_programs_scope"] == (
        "CLOSED_BOUNDARY_BUCKET_ONLY"
    )
    assert replay["independence_scope"] == (
        "INDEPENDENT_OF_ENDPOINT_REPORTED_WITNESS_NOT_A_THIRD_IMPLEMENTATION"
    )
    assert runtime["container_network"] == "none"
    assert runtime["container_pull_policy"] == "never"
    assert runtime["network_access_performed"] is False
    assert runtime["target_volume_removed_after_run"] is True
    routing = value["routing"]
    assert isinstance(routing, dict)
    assert routing == {
        "authority": "ENGINEERING_ONLY",
        "formal_status_promotion_allowed": False,
        "from_max_top_level_clauses": 3,
        "operation": "reduce max_top_level_clauses from 3 to 2",
        "preregistered_shrink_order_step": 4,
        "to_max_top_level_clauses": 2,
    }
    assert value["required_next_action"] == (
        "BEGIN_PREREGISTERED_SHRINK_STEP_4_ENGINEERING_REDUCE_MAX_TOP_LEVEL_"
        "CLAUSES_FROM_3_TO_2_WITHOUT_FORMAL_STATUS_PROMOTION"
    )
