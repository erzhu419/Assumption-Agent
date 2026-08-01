from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from replication_runtime.gscl_scar_cssm_v1 import worker


_H = "a" * 64
_GPU = "GPU-32d6e292-70cd-50a0-405b-e344d2da8d39"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _complete_evidence() -> dict[str, object]:
    resource = {
        "leaf_call_count": 2,
        "reported_success_candidate_count": 7,
        "reported_success_forward_batch_count": 3,
    }
    side = {
        "binder": {},
        "bounded_set": {},
        "document_envelope": {
            "leaf_records": [{}, {}],
            "receipt": {"receipt": {"resource_summary": resource}},
        },
        "slot_graph": {},
    }
    mapping = {
        "semantic_mapping": {},
        "structural_mapping": {},
        "target_color_shuffle_mapping": {},
    }
    return {
        "availability": "COMPLETE",
        "error_code": None,
        "semantic_matrix": {},
        "sides": {"left": side, "right": side},
        "variants": {"base": mapping, "system_swap": mapping},
    }


def _formed() -> dict[str, object]:
    return {
        "diagnostics": {},
        "execution": {
            "document_call_count": 2,
            "error_code": None,
            "structural_status": "EXECUTED_WITHOUT_TYPED_FAILURE",
        },
        "item_token": "scar-item-v1-" + "b" * 64,
        "private_mechanism_receipts": _complete_evidence(),
        "proposal_pools": {},
        "variants": {},
    }


def _patch_source_free_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        worker,
        "_require_output_root",
        lambda _: {"filesystem_type": "ext4"},
    )
    monkeypatch.setattr(
        worker,
        "_implementation_closure",
        lambda: {"self_sha256": _H},
    )
    monkeypatch.setattr(
        worker,
        "_load_sandbox_receipt",
        lambda *args, **kwargs: ({"self_sha256": _H}, {"sha256": _H}),
    )
    monkeypatch.setattr(worker, "_process_status", lambda: {"ok": True})
    monkeypatch.setattr(
        worker, "_network_family_negative_canary", lambda: {"ok": True}
    )
    monkeypatch.setattr(
        worker, "_forbidden_file_negative_canary", lambda _: {"ok": True}
    )
    monkeypatch.setattr(
        worker,
        "_require_one_visible_cuda_device",
        lambda **kwargs: {"physical_uuid": _GPU},
    )
    monkeypatch.setattr(
        worker,
        "_load_qwen_runtime",
        lambda **kwargs: (object(), {"runtime": "qwen"}),
    )
    monkeypatch.setattr(
        worker,
        "_load_minilm",
        lambda **kwargs: (object(), _H, {"runtime": "minilm"}),
    )
    monkeypatch.setattr(
        worker,
        "_runtime_receipt",
        lambda **kwargs: {"self_sha256": _H},
    )
    monkeypatch.setattr(
        worker,
        "_wait_for_action_release",
        lambda *args, **kwargs: (
            {"self_sha256": _H},
            {"sha256": _H, "size_bytes": 1},
        ),
    )


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "out"
    root.mkdir(mode=0o700)
    action_pack = {
        "action_commitment_sha256": _H,
        "items": [{"item_token": "unused"}],
    }
    action_file = {"sha256": _H, "size_bytes": 1}
    action_open_observation: dict[str, bool] = {}

    def load_action(*args, **kwargs):
        action_open_observation["runtime"] = (root / "shard0.runtime.safe.json").exists()
        action_open_observation["sentinel"] = (root / "shard0.attempt.sentinel").exists()
        return action_pack, action_file

    monkeypatch.setattr(worker, "_load_action_pack", load_action)
    monkeypatch.setattr(
        worker.action,
        "form_scar_cssm_item_action_v1",
        lambda *args, **kwargs: _formed(),
    )
    terminal = worker.run_shard(
        action_pack_path=tmp_path / "action.json",
        output_root=root,
        study_id="study",
        shard_index=0,
        qwen_model_root=tmp_path / "qwen",
        qwen_manifest_path=tmp_path / "qwen.json",
        minilm_model_root=tmp_path / "minilm",
        minilm_manifest_path=tmp_path / "minilm.json",
        sandbox_receipt_path=tmp_path / "sandbox.json",
        action_release_path=tmp_path / "release.json",
        forbidden_label_probe_path=tmp_path / "labels.json",
        expected_action_file_sha256=_H,
        expected_action_commitment_sha256=_H,
        expected_implementation_closure_sha256=_H,
        expected_sandbox_receipt_sha256=_H,
        expected_execution_freeze_sha256=_H,
        expected_gpu_uuid=_GPU,
        expected_peer_gpu_uuid=(
            "GPU-db2137c8-0f6b-b790-a698-6bfbbd5dc9eb"
        ),
    )
    return root, terminal, action_open_observation


def test_attempt_claim_follows_source_free_runtime_and_precedes_action_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_source_free_runtime(monkeypatch)
    root, terminal, observed = _run(tmp_path, monkeypatch)
    assert observed == {"runtime": True, "sentinel": True}
    assert terminal["status"] == "complete"
    assert terminal["mechanism_resource_totals"]["leaf_call_count"] == 4
    assert terminal["mechanism_resource_totals"]["candidate_count"] == 14
    assert terminal["mechanism_resource_totals"]["forward_batch_count"] == 6
    record = json.loads(
        (root / "shard0.records.private.jsonl").read_text().strip()
    )
    assert set(record) == {
        "evidence",
        "item_token",
        "ordinal_within_shard",
        "prediction",
        "self_sha256",
    }
    body = dict(record)
    self_hash = body.pop("self_sha256")
    assert self_hash == hashlib.sha256(_canonical(body)).hexdigest()
    assert "private_mechanism_receipts" not in record["prediction"]


def test_source_free_failure_never_opens_action_or_claims_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_source_free_runtime(monkeypatch)
    root = tmp_path / "out"
    root.mkdir(mode=0o700)
    opened = False

    def fail_if_opened(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("action pack was opened")

    monkeypatch.setattr(worker, "_load_action_pack", fail_if_opened)
    monkeypatch.setattr(
        worker,
        "_load_qwen_runtime",
        lambda **kwargs: (_ for _ in ()).throw(
            worker.ScarCssmWorkerError("WORKER_QWEN_RUNTIME_INVALID")
        ),
    )
    with pytest.raises(worker.ScarCssmWorkerError):
        worker.run_shard(
            action_pack_path=tmp_path / "action.json",
            output_root=root,
            study_id="study",
            shard_index=0,
            qwen_model_root=tmp_path / "qwen",
            qwen_manifest_path=tmp_path / "qwen.json",
            minilm_model_root=tmp_path / "minilm",
            minilm_manifest_path=tmp_path / "minilm.json",
            sandbox_receipt_path=tmp_path / "sandbox.json",
            action_release_path=tmp_path / "release.json",
            forbidden_label_probe_path=tmp_path / "labels.json",
            expected_action_file_sha256=_H,
            expected_action_commitment_sha256=_H,
            expected_implementation_closure_sha256=_H,
            expected_sandbox_receipt_sha256=_H,
            expected_execution_freeze_sha256=_H,
            expected_gpu_uuid=_GPU,
            expected_peer_gpu_uuid=(
                "GPU-db2137c8-0f6b-b790-a698-6bfbbd5dc9eb"
            ),
        )
    assert opened is False
    assert not (root / "shard0.attempt.sentinel").exists()


def test_private_reader_binds_inode_mode_link_count_and_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.json"
    path.write_bytes(b"{}")
    path.chmod(0o600)
    raw, receipt = worker._regular_private_file(path, maximum_bytes=4)
    assert raw == b"{}"
    assert receipt["sha256"] == hashlib.sha256(raw).hexdigest()
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(worker.ScarCssmWorkerError):
        worker._regular_private_file(alias, maximum_bytes=4)


def test_premodel_failure_has_zero_mechanism_resources() -> None:
    counts = worker._mechanism_resource_counts(
        {"availability": "PREMODEL_TYPED_FAILURE"}
    )
    assert set(counts.values()) == {0}


@pytest.mark.parametrize(
    "reported",
    (
        "32d6e292-70cd-50a0-405b-e344d2da8d39",
        "GPU-32d6e292-70cd-50a0-405b-e344d2da8d39",
    ),
)
def test_gpu_uuid_is_normalized_without_double_prefix(
    monkeypatch: pytest.MonkeyPatch, reported: str
) -> None:
    properties = SimpleNamespace(
        major=7,
        minor=5,
        name="NVIDIA GeForce RTX 2080",
        total_memory=8 * 1024**3,
        uuid=reported,
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: 1,
            is_available=lambda: True,
            current_device=lambda: 0,
            get_device_properties=lambda _: properties,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", _GPU)
    receipt = worker._require_one_visible_cuda_device(
        expected_gpu_uuid=_GPU
    )
    assert receipt["physical_uuid"] == _GPU


def test_memory_safe_qwen_evidence_uses_only_available_runtime_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declarations = {"loaded": "exact"}

    class _Parameter:
        device = "cuda:0"
        dtype = "torch.float16"
        shape = (2, 3)

    class _Model:
        def named_parameters(self):
            return (("weight", _Parameter()),)

    class _Runtime:
        _model = _Model()
        _source_sha256_value = _H
        strategy = "bounded_sparse_logits"
        runtime_commitment = _H

        def _validate_binding(self):
            return None

        def _loaded_declarations(self):
            return declarations

    runtime = _Runtime()
    manifest = SimpleNamespace(
        declarations=declarations,
        files=({"path": "model", "sha256": _H, "size": 1},),
        manifest_file_sha256=_H,
        runtime_requirements={"observed": "exact"},
        self_sha256=_H,
        tree_sha256=_H,
    )
    monkeypatch.setattr(
        worker.qwen_assets,
        "load_model_asset_manifest",
        lambda **kwargs: manifest,
    )
    monkeypatch.setattr(
        worker.document_qualification,
        "_verify_model_binding",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        worker.document_qualification,
        "_load_exact_runtime",
        lambda **kwargs: runtime,
    )
    monkeypatch.setattr(
        worker.document_qualification,
        "_validate_exact_runtime",
        lambda **kwargs: _H,
    )
    monkeypatch.setattr(
        worker.leaf_qualification,
        "_run_fixed_teacher_forced_canary",
        lambda _: {"self_sha256": _H},
    )
    monkeypatch.setattr(
        worker.document_qualification,
        "_validate_success_canary",
        lambda _: None,
    )
    monkeypatch.setattr(
        worker,
        "_observed_qwen_runtime_requirements",
        lambda _: {"observed": "exact"},
    )
    loaded, evidence = worker._load_qwen_runtime(
        model_root=Path("/model"), manifest_path=Path("/manifest")
    )
    assert loaded is runtime
    assert evidence["runtime_commitment"] == _H
    assert "runtime_receipt" not in evidence
    assert "target_double_run_receipt" not in evidence


def test_fixed_two_shard_partition_is_complete_and_disjoint() -> None:
    pack = {"items": [{"ordinal": index} for index in range(391)]}
    left = worker._selected_items(pack, shard_index=0)
    right = worker._selected_items(pack, shard_index=1)
    assert len(left) == 196
    assert len(right) == 195
    assert {row["ordinal"] for row in left}.isdisjoint(
        row["ordinal"] for row in right
    )
    assert {row["ordinal"] for row in (*left, *right)} == set(range(391))


def test_implementation_closure_contains_known_transitive_execution_files() -> None:
    required = {
        "assumption_agent/gscl_narrative_correspondence_v1.py",
        "assumption_agent/gscl_unit_mapping_v2.py",
        "assumption_agent/meta_assumption.py",
        "assumption_agent/universal_assumption_ontology_v1.py",
        "replication_runtime/gscl_narrative_extractor_v2/contract.py",
        "replication_runtime/gscl_narrative_extractor_v2/memory_safe_qwen.py",
    }
    assert required <= set(worker.IMPLEMENTATION_RELATIVE_PATHS)


def test_implementation_closure_covers_every_static_local_import() -> None:
    code_root = Path(worker.__file__).resolve().parents[2]
    frozen = set(worker.IMPLEMENTATION_RELATIVE_PATHS)

    def resolve(module: str) -> str | None:
        stem = code_root.joinpath(*module.split("."))
        if stem.with_suffix(".py").is_file():
            return str(stem.with_suffix(".py").relative_to(code_root))
        if (stem / "__init__.py").is_file():
            return str((stem / "__init__.py").relative_to(code_root))
        return None

    missing: set[str] = set()
    for relative in frozen:
        path = code_root / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        package_parts = list(Path(relative).with_suffix("").parts[:-1])
        if path.name == "__init__.py":
            package_parts = list(Path(relative).parts[:-1])
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    prefix = package_parts[: len(package_parts) - node.level + 1]
                    base = ".".join((*prefix, *((base,) if base else ())))
                candidates = [base] + [
                    f"{base}.{alias.name}" if base else alias.name
                    for alias in node.names
                ]
            for module in candidates:
                if module.startswith(("assumption_agent", "replication_runtime")):
                    resolved = resolve(module)
                    if resolved is not None and resolved not in frozen:
                        missing.add(resolved)
    assert missing == set()


def test_action_release_binds_both_distinct_gpu_runtime_receipts(
    tmp_path: Path,
) -> None:
    peer = "GPU-db2137c8-0f6b-b790-a698-6bfbbd5dc9eb"
    body = {
        "action_commitment_sha256": _H,
        "action_file_sha256": _H,
        "execution_freeze_sha256": _H,
        "gpu_uuid_by_shard": {"0": _GPU, "1": peer},
        "runtime_receipt_file_sha256_by_shard": {
            "0": _H,
            "1": "c" * 64,
        },
        "schema": worker.ACTION_RELEASE_SCHEMA,
        "shard_count": 2,
        "status": "release_both_shards_to_action_pack",
        "study_id": "study",
    }
    value = {**body, "self_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    path = tmp_path / "release.json"
    path.write_bytes(_canonical(value))
    path.chmod(0o600)
    receipt, file_receipt = worker._wait_for_action_release(
        path,
        study_id="study",
        shard_index=0,
        own_runtime_file_sha256=_H,
        expected_action_file_sha256=_H,
        expected_action_commitment_sha256=_H,
        expected_execution_freeze_sha256=_H,
        expected_gpu_uuid=_GPU,
        expected_peer_gpu_uuid=peer,
    )
    assert receipt["self_sha256"] == value["self_sha256"]
    assert file_receipt["sha256"] == hashlib.sha256(_canonical(value)).hexdigest()
    with pytest.raises(worker.ScarCssmWorkerError):
        worker._wait_for_action_release(
            path,
            study_id="study",
            shard_index=0,
            own_runtime_file_sha256=_H,
            expected_action_file_sha256=_H,
            expected_action_commitment_sha256=_H,
            expected_execution_freeze_sha256=_H,
            expected_gpu_uuid=_GPU,
            expected_peer_gpu_uuid=_GPU,
        )


def test_sandbox_receipt_loader_returns_validated_receipt_and_file_binding(
    tmp_path: Path,
) -> None:
    body = {
        "action_external_network_denied": True,
        "action_label_path_denied": True,
        "ip_address_deny": "any",
        "restrict_address_families": "AF_UNIX",
        "schema": worker.SANDBOX_RECEIPT_SCHEMA,
        "status": "frozen",
        "study_id": "study",
    }
    value = {**body, "self_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    raw = _canonical(value)
    path = tmp_path / "sandbox.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    receipt, file_receipt = worker._load_sandbox_receipt(
        path,
        expected_file_sha256=hashlib.sha256(raw).hexdigest(),
        study_id="study",
    )
    assert receipt == value
    assert file_receipt["sha256"] == hashlib.sha256(raw).hexdigest()
