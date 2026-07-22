from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import pytest

from replication_runtime.tatqa_p20_v1 import runtime_attestation_v1 as subject


PROJECT = Path(__file__).parents[1]
MANIFEST = PROJECT / "manifests/tatqa_p19_hipporag_runtime_attestation_v1.json"
BUILDER = PROJECT / "replication_runtime/tatqa_p20_v1/runtime_attestation_v1.py"
P19_BUILDER = PROJECT / "replication_runtime/tatqa_p19_v1/runtime_attestation_v1.py"
MANIFEST_FILE_SHA256 = (
    "96479f597bbf6ae9f69998df375816db9d870634d787976513ccb5bbef173955"
)
MANIFEST_RECEIPT_SHA256 = (
    "f12863b59a83e19188ccbf35208cafdf2b7c857daf404749a58e7f7787a07618"
)
BUILDER_FILE_SHA256 = (
    "8344353e326e0c5f986bd29a6aea65903a1271c3444e8ca95d372317a072be07"
)
REMOTE = "/home/erzhu419/p17_all_remote_20260722/runtime/reconstruction_v2"


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text("ascii"))


def test_actual_manifest_and_builder_are_exactly_bound() -> None:
    raw = MANIFEST.read_bytes()
    value = _manifest()
    assert hashlib.sha256(raw).hexdigest() == MANIFEST_FILE_SHA256
    assert value["receipt_sha256"] == MANIFEST_RECEIPT_SHA256
    assert raw == subject.canonical_json_bytes(value)
    assert subject.validate_receipt_structure(value) == value
    assert hashlib.sha256(BUILDER.read_bytes()).hexdigest() == BUILDER_FILE_SHA256
    assert BUILDER.read_bytes() == P19_BUILDER.read_bytes()
    assert value["implementation_binding"] == {
        "module_file_sha256": BUILDER_FILE_SHA256,
        "module_size_bytes": BUILDER.stat().st_size,
        "schema": "tatqa_p19_hipporag_runtime_attestation_builder_v1",
    }


def test_manifest_binds_the_hardened_source_and_both_actual_models() -> None:
    assets = _manifest()["asset_bindings"]
    assert assets["HippoRAG_source"] == {
        "contains_same_host_nonportable_bytecode": True,
        "nonportable_bytecode_file_count": 36,
        "root_role": "direct_hardened_P17_runtime_source",
        "tree": subject.EXPECTED_SOURCE_TREE,
    }
    assert assets["SmolLM"]["tree"] == subject.EXPECTED_SMOLLM_TREE
    assert assets["SmolLM"]["model_id"] == subject.SMOLLM_REPO_ID
    assert assets["SmolLM"]["revision"] == subject.SMOLLM_REVISION
    assert assets["MiniLM"]["generic_tree"] == (
        subject.EXPECTED_MINILM_GENERIC_TREE
    )
    assert assets["MiniLM"]["normative_tree_sha256"] == (
        subject.EXPECTED_MINILM_NORMATIVE_TREE_SHA256
    )
    assert assets["MiniLM"]["weights_sha256"] == (
        subject.EXPECTED_MINILM_WEIGHTS_SHA256
    )


def test_manifest_does_not_claim_the_old_v3_topology() -> None:
    value = _manifest()
    assert value["schema"] == "tatqa_p19_hipporag_runtime_attestation_v1"
    assert value["topology_decision"] == {
        "inherits_or_claims_MuSiQue_v3_topology": False,
        "old_MuSiQue_v3_official_source_tree_used": False,
        "source_identity": "complete_a644_hardened_P17_tree",
    }
    source = BUILDER.read_text("utf-8")
    assert "runtime_attestation_v3" not in source
    assert "30941a14e8dc48f7a41f8679ce6cba0bac9e3cdd99ed919560b45872e1058700" not in source


def test_active_interpreter_and_package_metadata_are_not_implicit() -> None:
    value = _manifest()
    runtime = value["runtime_python_binding"]
    assert runtime["lexical_path"] == (
        f"{REMOTE}/artifacts/bright_reasoning_retrieval_runtime_v1/"
        "hipporag_venv/bin/python"
    )
    assert runtime["resolved_path"] == (
        "/home/erzhu419/p17_all_remote_20260722/runtime/python310/bin/python3.10"
    )
    assert runtime["resolved_target_sha256"] == (
        subject.EXPECTED_RUNTIME_PYTHON_TARGET_SHA256
    )
    assert runtime["pyvenv_cfg_sha256"] == subject.EXPECTED_PYVENV_CFG_SHA256
    assert runtime["python_version"] == "3.10.12"
    assert runtime["samefile_with_active_sys_executable"] is True
    rows = value["active_distribution_binding"]["rows"]
    assert {row["name"]: row["version"] for row in rows} == (
        subject.EXPECTED_ACTIVE_DISTRIBUTION_VERSIONS
    )
    assert value["active_distribution_binding"]["set_sha256"] == (
        subject.stable_hash(rows)
    )
    for row in rows:
        if row["version"] is None:
            assert row["dist_info_tree_sha256"] is None
            assert row["dist_info_file_count"] == 0
        else:
            assert row["dist_info_file_count"] > 0
            assert len(row["dist_info_tree_sha256"]) == 64


def test_active_hipporag_origin_is_inside_the_a644_tree() -> None:
    rows = _manifest()["active_module_binding"]["rows"]
    assert _manifest()["active_module_binding"]["set_sha256"] == (
        subject.stable_hash(rows)
    )
    hippo = next(row for row in rows if row["distribution_name"] == "hipporag")
    assert hippo["origin_path"] == (
        f"{REMOTE}/reference/self_evo_continual_20260707/repos/"
        "HippoRAG/src/hipporag/__init__.py"
    )
    assert len(hippo["origin_file_sha256"]) == 64


def test_receipt_exposes_the_required_dual_interpreter_decision() -> None:
    decision = _manifest()["compatibility_decision"]
    assert (
        decision["P17_HippoRAG_interpreter_is_exact_QASPER_controller_runtime"]
        is False
    )
    assert decision["decision"] == (
        "reuse_for_HippoRAG_only_and_require_a_separate_exact_QASPER_controller_interpreter"
    )
    assert decision["qasper_required_versions"] == (
        subject.QASPER_CONTROLLER_REQUIRED_VERSIONS
    )
    mismatches = decision["mismatch_rows"]
    assert {row["name"] for row in mismatches} == set(
        subject.QASPER_CONTROLLER_REQUIRED_VERSIONS
    )


def test_receipt_is_source_free_and_secret_free() -> None:
    scope = _manifest()["source_free_scope"]
    assert scope == {
        "api_or_online_evaluator_calls": 0,
        "environment_variable_names_or_values_recorded": False,
        "external_network_calls": 0,
        "formal_TAT_QA_source_or_rows_accessed": False,
        "model_inference_calls": 0,
        "provider_or_API_credentials_read": False,
    }
    raw = MANIFEST.read_text("ascii").casefold()
    for forbidden in ("sk-", "api_key", "tatqa_dataset", "train.json", "dev.json"):
        assert forbidden not in raw


def test_structure_rejects_any_self_hash_tamper() -> None:
    value = _manifest()
    value["status"] = "tampered"
    with pytest.raises(subject.TatqaP19RuntimeAttestationError, match="self hash"):
        subject.validate_receipt_structure(value)


def test_tree_receipt_is_complete_and_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.bin").write_bytes(b"a")
    first = subject.tree_receipt(root, "test tree")
    (root / "a.bin").write_bytes(b"b")
    assert subject.tree_receipt(root, "test tree") != first
    (root / "link").symlink_to(root / "a.bin")
    with pytest.raises(subject.TatqaP19RuntimeAttestationError, match="non-regular"):
        subject.tree_receipt(root, "test tree")


def test_exclusive_writer_reopens_canonical_receipt() -> None:
    value = _manifest()
    with tempfile.TemporaryDirectory(
        prefix="tatqa-p19-attestation-", dir="/dev/shm"
    ) as directory:
        output = Path(directory) / "receipt.json"
        assert subject.write_attestation_exclusive(output, value) == (
            MANIFEST_FILE_SHA256
        )
        assert subject.load_attestation(output) == value
        with pytest.raises(subject.TatqaP19RuntimeAttestationError, match="consumed"):
            subject.write_attestation_exclusive(output, value)
