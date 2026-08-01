from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from replication_runtime.gscl_minilm_portable_v1 import (
    target_qualification as qualification,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    model = root / "model"
    model.mkdir(mode=0o700)
    asset = root / "asset.json"
    asset.write_bytes(b'{"asset":"public"}\n')
    asset.chmod(0o600)
    return {
        "root": root,
        "model": model,
        "asset": asset,
        "bundle": root / "target_bundle",
    }


def _fake_writer(**kwargs: object) -> dict[str, object]:
    target = Path(kwargs["target_manifest_path"])
    body = {
        "base_asset": {
            "model_tree_sha256": "1" * 64,
        },
        "public_synthetic_canary": {
            "target_observed_float32_sha256": "2" * 64,
            "target_observed_quantized_sha256": "3" * 64,
        },
        "schema": qualification.GSCL_MINILM_TARGET_SCHEMA,
    }
    value = {
        **body,
        "self_sha256": hashlib.sha256(_canonical(body)).hexdigest(),
    }
    raw = _canonical(value) + b"\n"
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {
        "target_manifest_file_sha256": hashlib.sha256(
            raw
        ).hexdigest(),
        "target_manifest_self_sha256": value["self_sha256"],
    }


def test_source_free_target_qualification_publishes_canonical_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        qualification,
        "write_target_manifest_qualification_only",
        _fake_writer,
    )
    receipt = qualification.run_source_free_target_qualification(
        asset_manifest_path=paths["asset"],
        model_root=paths["model"],
        output_bundle_path=paths["bundle"],
    )
    target = paths["bundle"] / "target_manifest.json"
    safe_receipt = paths["bundle"] / "qualification.safe.json"
    raw = safe_receipt.read_bytes()
    assert raw == _canonical(receipt) + b"\n"
    assert receipt["status"] == (
        "PASS_MINILM_TARGET_SOURCE_FREE_QUALIFICATION"
    )
    assert receipt["official_source_open_count"] == 0
    assert receipt["label_open_count"] == 0
    assert receipt["network_call_count"] == 0
    assert receipt["formal_measurement"] is False
    assert receipt["effect_gate_added"] is False
    assert receipt["minilm_model_construction_count"] == 1
    assert (paths["bundle"].stat().st_mode & 0o777) == 0o700
    assert (target.stat().st_mode & 0o777) == 0o600
    assert (safe_receipt.stat().st_mode & 0o777) == 0o600


def test_complete_existing_bundle_recovers_without_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        qualification,
        "write_target_manifest_qualification_only",
        _fake_writer,
    )
    expected = qualification.run_source_free_target_qualification(
        asset_manifest_path=paths["asset"],
        model_root=paths["model"],
        output_bundle_path=paths["bundle"],
    )
    calls = 0

    def forbidden_writer(**_: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise AssertionError("must not construct model")

    monkeypatch.setattr(
        qualification,
        "write_target_manifest_qualification_only",
        forbidden_writer,
    )
    observed = qualification.run_source_free_target_qualification(
        asset_manifest_path=paths["asset"],
        model_root=paths["model"],
        output_bundle_path=paths["bundle"],
    )
    assert calls == 0
    assert observed == expected


def test_receipt_publish_failure_never_exposes_final_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        qualification,
        "write_target_manifest_qualification_only",
        _fake_writer,
    )

    def fail_publish(_: Path, __: bytes) -> str:
        raise qualification.TargetQualificationError(
            "injected_receipt_failure"
        )

    monkeypatch.setattr(
        qualification, "_publish_safe_once", fail_publish
    )
    with pytest.raises(
        qualification.TargetQualificationError,
        match="injected_receipt_failure",
    ):
        qualification.run_source_free_target_qualification(
            asset_manifest_path=paths["asset"],
            model_root=paths["model"],
            output_bundle_path=paths["bundle"],
        )
    assert not paths["bundle"].exists()
    assert not list(
        paths["root"].glob(".target_bundle.pending-*")
    )


def test_base_exception_after_target_write_never_exposes_final_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path)

    def interrupted_writer(**kwargs: object) -> dict[str, object]:
        _fake_writer(**kwargs)
        raise SystemExit(143)

    monkeypatch.setattr(
        qualification,
        "write_target_manifest_qualification_only",
        interrupted_writer,
    )
    with pytest.raises(SystemExit, match="143"):
        qualification.run_source_free_target_qualification(
            asset_manifest_path=paths["asset"],
            model_root=paths["model"],
            output_bundle_path=paths["bundle"],
        )
    assert not paths["bundle"].exists()
    assert not list(
        paths["root"].glob(".target_bundle.pending-*")
    )


def test_bundle_is_hidden_until_both_files_are_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        qualification,
        "write_target_manifest_qualification_only",
        _fake_writer,
    )
    real_rename = qualification._rename_bundle_no_replace
    observations: list[tuple[bool, set[str]]] = []

    def observe_then_rename(source: Path, destination: Path) -> None:
        observations.append(
            (destination.exists(), set(os.listdir(source)))
        )
        real_rename(source, destination)

    monkeypatch.setattr(
        qualification,
        "_rename_bundle_no_replace",
        observe_then_rename,
    )
    qualification.run_source_free_target_qualification(
        asset_manifest_path=paths["asset"],
        model_root=paths["model"],
        output_bundle_path=paths["bundle"],
    )
    assert observations == [
        (
            False,
            {"target_manifest.json", "qualification.safe.json"},
        )
    ]
    assert paths["bundle"].is_dir()


def test_invalid_existing_bundle_fails_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path)
    paths["bundle"].mkdir(mode=0o700)
    paths["bundle"].chmod(0o700)
    calls = 0

    def forbidden_writer(**_: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise AssertionError("must not construct model")

    monkeypatch.setattr(
        qualification,
        "write_target_manifest_qualification_only",
        forbidden_writer,
    )
    with pytest.raises(
        qualification.TargetQualificationError,
        match="qualification_bundle_invalid",
    ):
        qualification.run_source_free_target_qualification(
            asset_manifest_path=paths["asset"],
            model_root=paths["model"],
            output_bundle_path=paths["bundle"],
        )
    assert calls == 0
