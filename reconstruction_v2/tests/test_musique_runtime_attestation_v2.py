from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import pytest

from assumption_agent.models import stable_hash
from assumption_agent.benchmarks.musique_formal_runtime_binding_v2 import (
    ADAPTER_ID,
    prepare_formal_runtime_v2,
    validate_formal_runtime_binding_v2,
)
import replication_runtime.musique_official_hipporag_v1.adapter_v2 as adapter_v2
import replication_runtime.musique_official_hipporag_v1.runtime_attestation_v2 as attestation
from replication_runtime.musique_official_hipporag_v1.binding import DEPENDENCY_NAMES
from replication_runtime.musique_official_hipporag_v1.contract import (
    MuSiQueOfficialHippoRAGError,
)


PROJECT = Path(__file__).parents[1]
BASE_BINDING = PROJECT / "manifests/musique_official_hipporag_retrieve_only_binding_v1.json"


def _safe_synthetic() -> dict[str, Any]:
    return {
        "benchmark_rows_read": 0,
        "candidate_count": 5,
        "external_network_transport_possible": False,
        "fixture_sha256": "1" * 64,
        "network_namespace_isolated": True,
        "official_core_index_called": True,
        "official_core_retrieve_called": True,
        "output_idx_count": 5,
        "output_idx_sha256": "2" * 64,
        "scores_computed": 0,
        "status": "passed_non_scoring_synthetic_local_retrieve_only",
    }


@dataclass
class _RuntimeFixture:
    runtime_python: Path
    llm: Path
    embedding: Path
    source: Path
    dependency_metadata: Path
    versions: dict[str, str | None]
    base: dict[str, Any]


def _runtime_fixture(tmp_path: Path) -> _RuntimeFixture:
    venv = tmp_path / "venv"
    runtime_python = venv / "bin" / "python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.write_bytes(b"synthetic-python-target-v1")
    os.chmod(runtime_python, 0o755)
    base_home = tmp_path / "base" / "bin"
    base_home.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        "\n".join(
            (
                f"home = {base_home}",
                "include-system-site-packages = false",
                "version = 3.11.9",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    site_packages = venv / "lib" / "python3.11" / "site-packages"
    site_packages.mkdir(parents=True)
    versions: dict[str, str | None] = {
        name: (None if name == "vllm" else f"1.0.{ordinal}")
        for ordinal, name in enumerate(DEPENDENCY_NAMES)
    }
    dependency_metadata = Path()
    for name in DEPENDENCY_NAMES:
        version = versions[name]
        if version is None:
            continue
        dist_name = name.replace("-", "_")
        dist_info = site_packages / f"{dist_name}-{version}.dist-info"
        dist_info.mkdir()
        metadata = dist_info / "METADATA"
        metadata.write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
            encoding="utf-8",
        )
        (dist_info / "RECORD").write_text("synthetic-record\n", encoding="utf-8")
        if name == "openai":
            dependency_metadata = metadata
    source = site_packages / "hipporag" / "core.py"
    source.parent.mkdir()
    source.write_text("class HippoRAG:\n    pass\n", encoding="utf-8")
    llm = tmp_path / "llm"
    embedding = tmp_path / "embedding"
    llm.mkdir()
    embedding.mkdir()
    (llm / "weights.bin").write_bytes(b"llm-weights-v1")
    (embedding / "weights.bin").write_bytes(b"embedding-weights-v1")
    snapshot = attestation._filesystem_snapshot(
        runtime_python=runtime_python,
        local_llm_model=llm,
        local_embedding_model=embedding,
        expected_versions=versions,
    )
    base = {
        "asset_binding": {
            "local_embedding_asset_sha256": snapshot[
                "local_embedding_asset_sha256"
            ],
            "local_llm_asset_sha256": snapshot["local_llm_asset_sha256"],
        },
        "implementation_binding": {},
        "official_source_binding": {
            "python_source_file_count": snapshot["official_source_file_count"],
            "python_source_tree_sha256": snapshot["official_source_tree_sha256"],
        },
        "qualification_binding": {
            "path": "manifests/official_hipporag_runtime_adapter_qualification_v1.json",
            "qualification_sha256": "3" * 64,
        },
        "receipt_sha256": "4" * 64,
        "runtime_binding": {
            "dependency_versions": versions,
            "pyvenv_cfg_sha256": snapshot["pyvenv_cfg_sha256"],
            "runtime_python_target_sha256": snapshot[
                "runtime_python_target_sha256"
            ],
        },
        "schema": "synthetic-v1-base-binding",
        "synthetic_local_qualification": _safe_synthetic(),
    }
    return _RuntimeFixture(
        runtime_python=runtime_python,
        llm=llm,
        embedding=embedding,
        source=source,
        dependency_metadata=dependency_metadata,
        versions=versions,
        base=base,
    )


def _build_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_RuntimeFixture, Path, dict[str, int]]:
    fixture = _runtime_fixture(tmp_path)
    counts = {"probe": 0, "synthetic": 0}

    def fake_base(_path: Path, *, project_root: Path):
        assert project_root == PROJECT
        return fixture.base, "5" * 64

    def fake_probe(_runtime_python: Path) -> dict[str, Any]:
        counts["probe"] += 1
        return {"qualified": True}

    def fake_runtime_from_probe(
        probe: Mapping[str, Any], qualification: Mapping[str, Any]
    ) -> dict[str, Any]:
        assert probe == {"qualified": True}
        assert qualification["qualified"] is True
        return fixture.base["runtime_binding"]

    def fake_launch(**kwargs: Any) -> None:
        counts["synthetic"] += 1
        kwargs["output_path"].write_bytes(b"[0,1,2,3,4]\n")

    monkeypatch.setattr(attestation, "_base_binding", fake_base)
    monkeypatch.setattr(attestation, "_runtime_probe", fake_probe)
    monkeypatch.setattr(attestation, "_runtime_binding_from_probe", fake_runtime_from_probe)
    monkeypatch.setattr(attestation, "_launch_worker", fake_launch)
    receipt = attestation.qualify_and_build_attestation_v2(
        project_root=PROJECT,
        base_binding_receipt_path=BASE_BINDING,
        runtime_python=fixture.runtime_python,
        local_llm_model=fixture.llm,
        local_embedding_model=fixture.embedding,
    )
    path = tmp_path / "attestation.v2.json"
    path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return fixture, path, counts


def test_pre_freeze_is_exactly_one_shot_and_formal_entry_never_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, receipt_path, counts = _build_receipt(tmp_path, monkeypatch)
    assert counts == {"probe": 1, "synthetic": 1}
    attestation._ATTESTATION_CACHE.clear()

    def forbidden_probe(_runtime_python: Path) -> dict[str, Any]:
        raise AssertionError("formal entry attempted an executable identity probe")

    monkeypatch.setattr(attestation, "_runtime_probe", forbidden_probe)
    result = attestation.verify_formal_runtime_attestation_v2(
        project_root=PROJECT,
        attestation_receipt_path=receipt_path,
        base_binding_receipt_path=BASE_BINDING,
        runtime_python=fixture.runtime_python,
        local_llm_model=fixture.llm,
        local_embedding_model=fixture.embedding,
    )
    assert result["formal_entry_executable_identity_probe_calls"] == 0


@pytest.mark.parametrize(
    "tamper",
    ["venv_python", "venv_metadata", "official_source", "llm_asset", "embedding_asset"],
)
def test_formal_filesystem_attestation_fails_closed_on_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    fixture, receipt_path, _counts = _build_receipt(tmp_path, monkeypatch)
    if tamper == "venv_python":
        fixture.runtime_python.write_bytes(b"tampered-python-target")
        os.chmod(fixture.runtime_python, 0o755)
    elif tamper == "venv_metadata":
        fixture.dependency_metadata.write_text(
            fixture.dependency_metadata.read_text(encoding="utf-8") + "Tampered: yes\n",
            encoding="utf-8",
        )
    elif tamper == "official_source":
        fixture.source.write_text("raise RuntimeError('tampered')\n", encoding="utf-8")
    elif tamper == "llm_asset":
        (fixture.llm / "weights.bin").write_bytes(b"tampered-llm")
    else:
        (fixture.embedding / "weights.bin").write_bytes(b"tampered-embedding")
    attestation._ATTESTATION_CACHE.clear()
    with pytest.raises(
        MuSiQueOfficialHippoRAGError,
        match="filesystem|pre-freeze qualification",
    ):
        attestation.verify_formal_runtime_attestation_v2(
            project_root=PROJECT,
            attestation_receipt_path=receipt_path,
            base_binding_receipt_path=BASE_BINDING,
            runtime_python=fixture.runtime_python,
            local_llm_model=fixture.llm,
            local_embedding_model=fixture.embedding,
        )


def test_pre_freeze_probe_failure_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _runtime_fixture(tmp_path)
    counts = {"probe": 0, "synthetic": 0}

    monkeypatch.setattr(
        attestation,
        "_base_binding",
        lambda _path, *, project_root: (fixture.base, "5" * 64),
    )

    def fail_once(_runtime_python: Path) -> dict[str, Any]:
        counts["probe"] += 1
        raise MuSiQueOfficialHippoRAGError("synthetic one-shot probe failure")

    def fake_launch(**kwargs: Any) -> None:
        counts["synthetic"] += 1
        kwargs["output_path"].write_bytes(b"[0,1,2,3,4]\n")

    monkeypatch.setattr(attestation, "_runtime_probe", fail_once)
    monkeypatch.setattr(attestation, "_launch_worker", fake_launch)
    with pytest.raises(MuSiQueOfficialHippoRAGError, match="one-shot probe failure"):
        attestation.qualify_and_build_attestation_v2(
            project_root=PROJECT,
            base_binding_receipt_path=BASE_BINDING,
            runtime_python=fixture.runtime_python,
            local_llm_model=fixture.llm,
            local_embedding_model=fixture.embedding,
        )
    assert counts == {"probe": 1, "synthetic": 1}


def test_pre_freeze_synthetic_worker_failure_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _runtime_fixture(tmp_path)
    counts = {"probe": 0, "synthetic": 0}
    monkeypatch.setattr(
        attestation,
        "_base_binding",
        lambda _path, *, project_root: (fixture.base, "5" * 64),
    )

    def fail_worker(**_kwargs: Any) -> None:
        counts["synthetic"] += 1
        raise MuSiQueOfficialHippoRAGError("fixed synthetic worker failure")

    def probe(_runtime_python: Path) -> dict[str, Any]:
        counts["probe"] += 1
        return {"qualified": True}

    monkeypatch.setattr(attestation, "_launch_worker", fail_worker)
    monkeypatch.setattr(attestation, "_runtime_probe", probe)
    with pytest.raises(MuSiQueOfficialHippoRAGError, match="worker failure"):
        attestation.qualify_and_build_attestation_v2(
            project_root=PROJECT,
            base_binding_receipt_path=BASE_BINDING,
            runtime_python=fixture.runtime_python,
            local_llm_model=fixture.llm,
            local_embedding_model=fixture.embedding,
        )
    assert counts == {"probe": 0, "synthetic": 1}


def test_pre_freeze_builder_has_no_qualifier_or_result_injection_api() -> None:
    assert tuple(inspect.signature(attestation.qualify_and_build_attestation_v2).parameters) == (
        "project_root",
        "base_binding_receipt_path",
        "runtime_python",
        "local_llm_model",
        "local_embedding_model",
    )


def test_v2_adapter_requires_explicit_attestation_and_launches_only_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"attest": 0, "worker": 0}
    manifests = tmp_path / "project" / "manifests"
    manifests.mkdir(parents=True)
    base = manifests / "base.json"
    receipt = manifests / "attestation.json"
    base.write_text("{}\n", encoding="utf-8")
    receipt.write_text("{}\n", encoding="utf-8")
    runtime = tmp_path / "venv" / "bin" / "python"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"python")
    os.chmod(runtime, 0o755)
    llm = tmp_path / "llm"
    embedding = tmp_path / "embedding"
    llm.mkdir()
    embedding.mkdir()
    (llm / "asset").write_bytes(b"llm")
    (embedding / "asset").write_bytes(b"embedding")

    def verify(**_kwargs: Any) -> dict[str, Any]:
        calls["attest"] += 1
        return {"formal_entry_executable_identity_probe_calls": 0}

    def launch(**kwargs: Any) -> None:
        calls["worker"] += 1
        kwargs["output_path"].write_bytes(b"[0,1,2,3,4]\n")

    monkeypatch.setattr(adapter_v2, "verify_formal_runtime_attestation_v2", verify)
    monkeypatch.setattr(adapter_v2, "_launch_worker", launch)
    result = adapter_v2.run_official_hipporag_retrieve_only_v2(
        question="Synthetic question?",
        paragraphs=[
            {
                "idx": idx,
                "title": f"Title {idx}",
                "paragraph_text": f"Text {idx}",
            }
            for idx in range(6)
        ],
        runtime_python=runtime,
        local_llm_model=llm,
        local_embedding_model=embedding,
        base_binding_receipt_path=base,
        attestation_receipt_path=receipt,
        work_root=tmp_path / "work",
    )
    assert result == (0, 1, 2, 3, 4)
    assert calls == {"attest": 1, "worker": 1}
    assert not (tmp_path / "work").exists()


def test_receipt_self_hash_covers_zero_probe_formal_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture, receipt_path, _counts = _build_receipt(tmp_path, monkeypatch)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["formal_entry_policy"]["executable_identity_probe_calls"] = 1
    receipt["receipt_sha256"] = stable_hash(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    attestation._ATTESTATION_CACHE.clear()
    with pytest.raises(MuSiQueOfficialHippoRAGError, match="formal-entry policy"):
        attestation.verify_formal_runtime_attestation_v2(
            project_root=PROJECT,
            attestation_receipt_path=receipt_path,
            base_binding_receipt_path=BASE_BINDING,
            runtime_python=_fixture.runtime_python,
            local_llm_model=_fixture.llm,
            local_embedding_model=_fixture.embedding,
        )


def test_runner_side_prepare_is_path_free_and_has_zero_entry_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, receipt_path, counts = _build_receipt(tmp_path, monkeypatch)
    attestation._ATTESTATION_CACHE.clear()

    def forbidden_probe(_runtime_python: Path) -> dict[str, Any]:
        raise AssertionError("runner-side formal entry attempted a probe")

    def forbidden_subprocess(*_args: Any, **_kwargs: Any):
        raise AssertionError("runner-side formal entry attempted a subprocess")

    monkeypatch.setattr(attestation, "_runtime_probe", forbidden_probe)
    monkeypatch.setattr(subprocess, "run", forbidden_subprocess)
    prepared = prepare_formal_runtime_v2(
        project_root=PROJECT,
        attestation_receipt_path=receipt_path,
        base_binding_receipt_path=BASE_BINDING,
        runtime_python=fixture.runtime_python,
        local_llm_model=fixture.llm,
        local_embedding_model=fixture.embedding,
    )
    assert counts == {"probe": 1, "synthetic": 1}
    assert prepared.safe_binding["adapter_id"] == ADAPTER_ID
    assert prepared.safe_binding["formal_entry_executable_identity_probe_calls"] == 0
    assert prepared.safe_binding["formal_entry_subprocess_calls"] == 0
    safe_raw = json.dumps(prepared.safe_binding, sort_keys=True)
    assert str(tmp_path) not in safe_raw
    assert validate_formal_runtime_binding_v2(prepared.safe_binding) == dict(
        prepared.safe_binding
    )


def test_prepared_runner_handle_calls_explicit_v2_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, receipt_path, _counts = _build_receipt(tmp_path, monkeypatch)
    attestation._ATTESTATION_CACHE.clear()
    prepared = prepare_formal_runtime_v2(
        project_root=PROJECT,
        attestation_receipt_path=receipt_path,
        base_binding_receipt_path=BASE_BINDING,
        runtime_python=fixture.runtime_python,
        local_llm_model=fixture.llm,
        local_embedding_model=fixture.embedding,
    )
    worker_calls: list[str] = []

    def launch(**kwargs: Any) -> None:
        worker_calls.append(kwargs["runtime_python"].name)
        kwargs["output_path"].write_bytes(b"[0,1,2,3,4]\n")

    monkeypatch.setattr(adapter_v2, "_launch_worker", launch)
    result = prepared.retrieve(
        question="Synthetic question?",
        paragraphs=[
            {
                "idx": idx,
                "title": f"Title {idx}",
                "paragraph_text": f"Text {idx}",
            }
            for idx in range(6)
        ],
        work_root=tmp_path / "prepared-work",
    )
    assert result == (0, 1, 2, 3, 4)
    assert worker_calls == ["python"]
    assert not (tmp_path / "prepared-work").exists()


def test_runner_binding_rejects_rehashed_nonzero_probe_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, receipt_path, _counts = _build_receipt(tmp_path, monkeypatch)
    attestation._ATTESTATION_CACHE.clear()
    prepared = prepare_formal_runtime_v2(
        project_root=PROJECT,
        attestation_receipt_path=receipt_path,
        base_binding_receipt_path=BASE_BINDING,
        runtime_python=fixture.runtime_python,
        local_llm_model=fixture.llm,
        local_embedding_model=fixture.embedding,
    )
    forged = dict(prepared.safe_binding)
    forged["formal_entry_executable_identity_probe_calls"] = 1
    forged["binding_sha256"] = stable_hash(
        {key: value for key, value in forged.items() if key != "binding_sha256"}
    )
    with pytest.raises(MuSiQueOfficialHippoRAGError, match="runtime policy"):
        validate_formal_runtime_binding_v2(forged)


def test_postflight_fresh_reverify_matches_unchanged_pre_run_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, receipt_path, _counts = _build_receipt(tmp_path, monkeypatch)
    attestation._ATTESTATION_CACHE.clear()
    prepared = prepare_formal_runtime_v2(
        project_root=PROJECT,
        attestation_receipt_path=receipt_path,
        base_binding_receipt_path=BASE_BINDING,
        runtime_python=fixture.runtime_python,
        local_llm_model=fixture.llm,
        local_embedding_model=fixture.embedding,
    )
    assert prepared.fresh_reverify() == prepared.safe_binding


def test_formal_prepare_bypasses_preexisting_cache_before_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, receipt_path, _counts = _build_receipt(tmp_path, monkeypatch)
    attestation._ATTESTATION_CACHE.clear()
    attestation.verify_formal_runtime_attestation_v2(
        project_root=PROJECT,
        attestation_receipt_path=receipt_path,
        base_binding_receipt_path=BASE_BINDING,
        runtime_python=fixture.runtime_python,
        local_llm_model=fixture.llm,
        local_embedding_model=fixture.embedding,
    )
    (fixture.embedding / "weights.bin").write_bytes(b"pre-authorization-tamper")

    with pytest.raises(
        MuSiQueOfficialHippoRAGError,
        match="filesystem|pre-freeze qualification",
    ):
        prepare_formal_runtime_v2(
            project_root=PROJECT,
            attestation_receipt_path=receipt_path,
            base_binding_receipt_path=BASE_BINDING,
            runtime_python=fixture.runtime_python,
            local_llm_model=fixture.llm,
            local_embedding_model=fixture.embedding,
        )


def test_cached_success_can_reuse_but_fresh_postflight_rejects_later_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, receipt_path, _counts = _build_receipt(tmp_path, monkeypatch)
    attestation._ATTESTATION_CACHE.clear()
    prepared = prepare_formal_runtime_v2(
        project_root=PROJECT,
        attestation_receipt_path=receipt_path,
        base_binding_receipt_path=BASE_BINDING,
        runtime_python=fixture.runtime_python,
        local_llm_model=fixture.llm,
        local_embedding_model=fixture.embedding,
    )
    (fixture.llm / "weights.bin").write_bytes(b"post-entry-tamper")

    cached = attestation.verify_formal_runtime_attestation_v2(
        project_root=PROJECT,
        attestation_receipt_path=receipt_path,
        base_binding_receipt_path=BASE_BINDING,
        runtime_python=fixture.runtime_python,
        local_llm_model=fixture.llm,
        local_embedding_model=fixture.embedding,
    )
    assert cached["attestation_receipt_sha256"] == prepared.safe_binding[
        "attestation_receipt_sha256"
    ]

    with pytest.raises(
        MuSiQueOfficialHippoRAGError,
        match="filesystem|pre-freeze qualification",
    ):
        prepared.fresh_reverify()

    # The failed fresh read evicts the stale cache; it cannot mask later checks.
    with pytest.raises(
        MuSiQueOfficialHippoRAGError,
        match="filesystem|pre-freeze qualification",
    ):
        attestation.verify_formal_runtime_attestation_v2(
            project_root=PROJECT,
            attestation_receipt_path=receipt_path,
            base_binding_receipt_path=BASE_BINDING,
            runtime_python=fixture.runtime_python,
            local_llm_model=fixture.llm,
            local_embedding_model=fixture.embedding,
        )


def test_cache_bypass_flag_is_exact_boolean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, receipt_path, _counts = _build_receipt(tmp_path, monkeypatch)
    with pytest.raises(MuSiQueOfficialHippoRAGError, match="must be boolean"):
        attestation.verify_formal_runtime_attestation_v2(
            project_root=PROJECT,
            attestation_receipt_path=receipt_path,
            base_binding_receipt_path=BASE_BINDING,
            runtime_python=fixture.runtime_python,
            local_llm_model=fixture.llm,
            local_embedding_model=fixture.embedding,
            bypass_cache=1,  # type: ignore[arg-type]
        )
