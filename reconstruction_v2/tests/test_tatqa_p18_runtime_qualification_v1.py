from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any

import pytest

from assumption_agent.benchmarks import tatqa_p18_acquisition_v1 as acquisition
from assumption_agent.benchmarks import tatqa_p18_runtime_qualification_v1 as qualification


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _semantic_hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).rstrip(b"\n")).hexdigest()


@pytest.fixture
def native_tmp_path() -> Path:
    # This module verifies POSIX 0600 receipts.  The repository-wide TMPDIR may
    # point at drvfs, whose synthesized mode bits cannot represent that check.
    path = Path(tempfile.mkdtemp(prefix="tatqa-p18-qualification-", dir="/tmp"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _arguments(tmp_path: Path) -> dict[str, object]:
    project = tmp_path / "project"
    project.mkdir()
    return {
        "project_root": project,
        "qualification_root": tmp_path / "qualification",
        "runtime_python": tmp_path / "runtime" / "bin" / "python",
        "qwen_model": tmp_path / "assets" / "qwen",
        "minilm_asset_manifest": tmp_path / "manifests" / "minilm.json",
        "minilm_model": tmp_path / "assets" / "minilm",
        "hippo_llm_model": tmp_path / "assets" / "hippo-llm",
        "hippo_embedding_model": tmp_path / "assets" / "hippo-embedding",
        "hipporag_source": tmp_path / "assets" / "hipporag-source",
        "hippo_attestation": tmp_path / "manifests" / "hippo.json",
        "runtime_implementation_commit": "a" * 40,
        "fingerprint_output": tmp_path / "evidence" / "fingerprint.json",
        "canary_output": tmp_path / "evidence" / "canary.json",
    }


def test_success_binds_exact_runtime_paths_fingerprint_and_production_canary(
    native_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = native_tmp_path
    arguments = _arguments(tmp_path)
    observed: dict[str, Any] = {}

    def fake_inventory(**kwargs: object) -> dict[str, object]:
        observed["inventory_arguments"] = kwargs
        return {"inventory": "fixed-public-runtime"}

    network = {
        "network_properties": [
            "IPAddressDeny=any",
            "RestrictAddressFamilies=AF_UNIX",
        ],
        "returncode": 0,
        "stdout_sha256": hashlib.sha256(b"").hexdigest(),
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
    }

    def fake_fingerprint(**kwargs: object) -> dict[str, object]:
        observed["fingerprint_arguments"] = kwargs
        body = {
            "schema": "test-runtime-fingerprint",
            "status": "verified_before_formal_source_open",
        }
        receipt = {**body, "self_sha256": _semantic_hash(body)}
        path = Path(kwargs["output_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical(receipt))
        return receipt

    class FakeEncoder:
        def __init__(self, paths: object) -> None:
            observed["encoder_paths"] = paths

    class FakeTypedRunner:
        def __init__(self, paths: object) -> None:
            observed["typed_paths"] = paths

        def abort_all_workers(self) -> tuple[()]:
            observed["typed_abort"] = True
            return ()

        def verify_all_workers_closed(self) -> tuple[()]:
            observed["typed_closed"] = True
            return ()

    class FakeHippoRunner:
        def __init__(self, paths: object) -> None:
            observed["hippo_paths"] = paths

        def abort_all_workers(self) -> tuple[()]:
            observed["hippo_abort"] = True
            return ()

        def verify_all_workers_closed(self) -> tuple[()]:
            observed["hippo_closed"] = True
            return ()

    def fake_canary(**kwargs: object) -> dict[str, object]:
        observed["canary_arguments"] = kwargs
        body = {
            "schema": "test-public-production-canary",
            "status": "qualified_before_formal_source_open",
            "hippo_canary_ran": True,
            "P1_retains_ordered_P0_top3": True,
            "P1_outside_P0_unit_count": 1,
            "typed_plan_worker_receipt_source": "capability_receipt_snapshot",
            "minilm_worker_receipt_source": "capability_receipt_snapshot",
            "hippo_worker_receipt_source": "capability_receipt_snapshot",
        }
        receipt = {**body, "self_sha256": _semantic_hash(body)}
        path = Path(kwargs["output_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_canonical(receipt))
        return receipt

    monkeypatch.setattr(qualification, "runtime_inventory", fake_inventory)
    monkeypatch.setattr(
        qualification.formal_runtime,
        "systemd_network_preflight",
        lambda: network,
    )
    monkeypatch.setattr(
        qualification.freeze, "build_runtime_fingerprint", fake_fingerprint
    )
    monkeypatch.setattr(
        qualification.formal_runtime,
        "verify_runtime_fingerprint",
        lambda paths: observed.setdefault("verified_paths", paths),
    )
    monkeypatch.setattr(
        qualification.formal_runtime, "BoundMiniLMEncoder", FakeEncoder
    )
    monkeypatch.setattr(
        qualification.formal_runtime,
        "SystemdTypedPlanBatchRunner",
        FakeTypedRunner,
    )
    monkeypatch.setattr(
        qualification.formal_runtime, "SystemdHippoByteRunner", FakeHippoRunner
    )
    monkeypatch.setattr(
        qualification.canary, "run_public_production_canary", fake_canary
    )
    monkeypatch.setattr(
        qualification.acquisition,
        "validate_production_canary_capability_receipts",
        lambda receipt: observed.setdefault("validated_canary", receipt),
    )

    terminal = qualification.run_runtime_qualification(**arguments)

    expected_inventory_arguments = {
        "runtime_python": Path(arguments["runtime_python"]).absolute(),
        "qwen_model": Path(arguments["qwen_model"]).absolute(),
        "minilm_manifest": Path(arguments["minilm_asset_manifest"]).absolute(),
        "hippo_attestation": Path(arguments["hippo_attestation"]).absolute(),
    }
    assert observed["inventory_arguments"] == expected_inventory_arguments

    fingerprint_arguments = observed["fingerprint_arguments"]
    assert fingerprint_arguments["runtime_inventory"] == {
        "inventory": "fixed-public-runtime"
    }
    assert fingerprint_arguments["systemd_network_preflight"] is network
    assert fingerprint_arguments["runtime_implementation_commit"] == "a" * 40
    assert fingerprint_arguments["asset_roots"] == {
        "Qwen": Path(arguments["qwen_model"]).absolute(),
        "MiniLM": Path(arguments["minilm_model"]).absolute(),
        "HippoRAG_LLM": Path(arguments["hippo_llm_model"]).absolute(),
        "HippoRAG_embedding": Path(
            arguments["hippo_embedding_model"]
        ).absolute(),
        "HippoRAG_source": Path(arguments["hipporag_source"]).absolute(),
    }

    paths = observed["verified_paths"]
    assert observed["encoder_paths"] is paths
    assert observed["typed_paths"] is paths
    assert observed["hippo_paths"] is paths
    assert paths.project_root == Path(arguments["project_root"]).resolve()
    assert paths.fingerprint_manifest == Path(
        arguments["fingerprint_output"]
    ).absolute()
    assert paths.work_root == Path(arguments["qualification_root"]) / "work"

    canary_arguments = observed["canary_arguments"]
    assert canary_arguments["runtime_fingerprint_path"] == paths.fingerprint_manifest
    assert canary_arguments["output_path"] == Path(
        arguments["canary_output"]
    ).absolute()
    assert canary_arguments["encoder"] is not None
    assert canary_arguments["typed_plan_runner"] is not None
    assert canary_arguments["hippo_runner"] is not None
    assert observed["validated_canary"]["hippo_canary_ran"] is True
    assert observed["typed_abort"] is observed["typed_closed"] is True
    assert observed["hippo_abort"] is observed["hippo_closed"] is True

    assert terminal["status"] == "qualified_before_formal_source_open"
    assert terminal["formal_source_opened"] is False
    assert terminal["runtime_fingerprint_self_sha256"]
    assert terminal["production_canary_self_sha256"]
    root = Path(arguments["qualification_root"])
    marker = json.loads((root / qualification.MARKER_FILENAME).read_text("ascii"))
    assert marker["runtime_implementation_commit"] == "a" * 40
    assert marker["formal_source_opened"] is False
    success_path = root / "qualification.terminal_success.json"
    assert json.loads(success_path.read_text("ascii")) == terminal
    assert stat.S_IMODE(success_path.stat().st_mode) == 0o600
    assert not (root / qualification.FAILURE_FILENAME).exists()


@pytest.mark.parametrize(
    "formal_relative",
    (
        acquisition.SOURCE_RECEIPT_RELATIVE,
        acquisition.SOURCE_ROOT_RELATIVE,
        acquisition.ACQUISITION_ROOT_RELATIVE,
    ),
)
def test_formal_source_or_acquisition_preexisting_refuses_before_claiming_root(
    native_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    formal_relative: Path,
) -> None:
    tmp_path = native_tmp_path
    arguments = _arguments(tmp_path)
    formal_path = Path(arguments["project_root"]) / formal_relative
    if formal_relative == acquisition.SOURCE_RECEIPT_RELATIVE:
        formal_path.parent.mkdir(parents=True, exist_ok=True)
        formal_path.write_bytes(b"never-opened-by-test")
    else:
        formal_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        qualification,
        "runtime_inventory",
        lambda **_kwargs: pytest.fail("runtime inventory must not execute"),
    )

    with pytest.raises(
        qualification.TatqaP18RuntimeQualificationError,
        match="formal source/acquisition state predates",
    ):
        qualification.run_runtime_qualification(**arguments)

    assert not Path(arguments["qualification_root"]).exists()
    assert not Path(arguments["fingerprint_output"]).exists()
    assert not Path(arguments["canary_output"]).exists()


@pytest.mark.parametrize("consumed", ("root", "fingerprint", "canary"))
def test_existing_one_shot_root_or_output_is_refused_before_runtime_work(
    native_tmp_path: Path, monkeypatch: pytest.MonkeyPatch, consumed: str
) -> None:
    tmp_path = native_tmp_path
    arguments = _arguments(tmp_path)
    if consumed == "root":
        target = Path(arguments["qualification_root"])
        target.mkdir()
        (target / "sentinel").write_text("preserve", encoding="ascii")
    else:
        key = f"{consumed}_output"
        target = Path(arguments[key])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("preserve", encoding="ascii")
    monkeypatch.setattr(
        qualification,
        "runtime_inventory",
        lambda **_kwargs: pytest.fail("runtime inventory must not execute"),
    )

    with pytest.raises(
        qualification.TatqaP18RuntimeQualificationError,
        match="already consumed",
    ):
        qualification.run_runtime_qualification(**arguments)

    if consumed == "root":
        assert (target / "sentinel").read_text("ascii") == "preserve"
    else:
        assert target.read_text("ascii") == "preserve"
    if consumed != "root":
        assert not Path(arguments["qualification_root"]).exists()


def test_failure_after_root_claim_writes_terminal_receipt_without_retry(
    native_tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp_path = native_tmp_path
    arguments = _arguments(tmp_path)

    def fail_inventory(**_kwargs: object) -> dict[str, object]:
        raise RuntimeError("synthetic offline inventory failure")

    monkeypatch.setattr(qualification, "runtime_inventory", fail_inventory)

    with pytest.raises(
        qualification.TatqaP18RuntimeQualificationError,
        match="failed terminally",
    ):
        qualification.run_runtime_qualification(**arguments)

    root = Path(arguments["qualification_root"])
    assert (root / qualification.MARKER_FILENAME).is_file()
    failure_path = root / qualification.FAILURE_FILENAME
    raw = failure_path.read_bytes()
    failure = json.loads(raw.decode("ascii"))
    assert raw == _canonical(failure)
    assert failure["status"] == (
        "terminal_no_retry_requalification_or_formal_source_open"
    )
    assert failure["failure_stage"] == "runtime_inventory"
    assert failure["formal_source_opened"] is False
    assert failure["external_network_calls_other_than_none"] == 0
    assert failure["api_or_online_evaluator_calls"] == 0
    body = dict(failure)
    declared = body.pop("self_sha256")
    assert declared == _semantic_hash(body)
    assert stat.S_IMODE(failure_path.stat().st_mode) == 0o600
    assert not Path(arguments["fingerprint_output"]).exists()
    assert not Path(arguments["canary_output"]).exists()
    assert not (root / "qualification.terminal_success.json").exists()
