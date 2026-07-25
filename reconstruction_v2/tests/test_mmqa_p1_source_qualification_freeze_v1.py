from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from assumption_agent.benchmarks import (
    mmqa_p1_source_qualification_freeze_v1 as f,
)
from assumption_agent.benchmarks import mmqa_p1_source_acquisition_v1 as a
from assumption_agent.benchmarks import mmqa_p1_source_qualification_v1 as q


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class _SyntheticProject:
    root: Path
    payloads: dict[str, bytes]
    receipt: dict[str, object]


def _git_blob_sha1(raw: bytes) -> str:
    digest = hashlib.sha1()  # nosec B324: synthetic Git object identity
    digest.update(f"blob {len(raw)}\0".encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def _write_private(path: Path, raw: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(raw)
    os.chmod(path, 0o600)


def _replace_receipt(
    synthetic: _SyntheticProject, body: dict[str, object]
) -> dict[str, object]:
    body = copy.deepcopy(body)
    body.pop("self_sha256", None)
    receipt = f._self_hashed(body)
    path = synthetic.root / f.DOWNLOAD_RECEIPT_RELATIVE
    path.unlink()
    _write_private(path, f._canonical_bytes(receipt))
    synthetic.receipt = receipt
    return receipt


@pytest.fixture
def synthetic_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _SyntheticProject:
    project = tmp_path / "reconstruction_v2"
    project.mkdir(mode=0o700)

    for relative in (
        f.CUSTODY_RELATIVE,
        f.DESIGN_RELATIVE,
        f.AUTHORIZATION_RELATIVE,
    ):
        destination = project / relative
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / relative).read_bytes())

    qualifier_raw = b"SYNTHETIC_FINAL_QUALIFIER_IMPLEMENTATION\n"
    test_raw = b"SYNTHETIC_FINAL_QUALIFIER_TESTS\n"
    _write_private(project / f.QUALIFIER_RELATIVE, qualifier_raw)
    _write_private(project / f.TEST_RELATIVE, test_raw)
    monkeypatch.setattr(
        f, "EXPECTED_QUALIFIER_SHA256", hashlib.sha256(qualifier_raw).hexdigest()
    )
    monkeypatch.setattr(
        f, "EXPECTED_TEST_SHA256", hashlib.sha256(test_raw).hexdigest()
    )

    source_root = project / f.SOURCE_ROOT_RELATIVE
    source_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    source_root.mkdir(mode=0o700)
    payloads = {
        file_name: (
            b"NOT_A_GZIP_OR_JSONL_SECRET_OPAQUE_BYTES_"
            + file_name.encode("ascii")
            + b"_"
            + bytes([ordinal]) * (ordinal + 5)
        )
        for ordinal, file_name in enumerate(f.FILE_ORDER, start=1)
    }
    contracts: dict[str, f.SourceFileBinding] = {}
    rows: list[dict[str, object]] = []
    for ordinal, file_name in enumerate(f.FILE_ORDER, start=1):
        raw = payloads[file_name]
        _write_private(source_root / file_name, raw)
        git_blob_sha1 = _git_blob_sha1(raw)
        contracts[file_name] = f.SourceFileBinding(
            file_name=file_name,
            expected_size_bytes=len(raw),
            expected_git_blob_sha1=git_blob_sha1,
        )
        rows.append(
            {
                "file_attempt_marker_sha256": hashlib.sha256(
                    f"attempt:{ordinal}:{file_name}".encode("ascii")
                ).hexdigest(),
                "file_name": file_name,
                "git_blob_sha1": git_blob_sha1,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    monkeypatch.setattr(f, "FORMAL_FILES", contracts)

    receipt_body: dict[str, object] = {
        "authorization_self_sha256": f.EXPECTED_AUTHORIZATION_SELF_SHA256,
        "completed_file_count": 4,
        "dataset_byte_decode_decompress_JSONL_or_row_parse_count": 0,
        "files": rows,
        "model_action_embedding_reranking_score_or_online_evaluator_count": 0,
        "network_attempt_count": 4,
        "network_attempt_count_per_file_maximum": 1,
        "nonmatching_host_redirect_count": 0,
        "one_shot_attempt_marker_sha256": "a" * 64,
        "response_body_or_URL_query_output_count": 0,
        "retry_resume_range_mirror_or_provider_switch_count": 0,
        "schema": "mmqa_p1_source_download_receipt_v1",
        "source_custody_self_sha256": f.EXPECTED_CUSTODY_SELF_SHA256,
        "source_root_relative": f.SOURCE_ROOT_RELATIVE.as_posix(),
        "status": (
            "four_fixed_sources_downloaded_identity_verified_not_parsed"
        ),
        "study_design_self_sha256": f.EXPECTED_DESIGN_SELF_SHA256,
        "study_id": f.STUDY_ID,
    }
    receipt = f._self_hashed(receipt_body)
    _write_private(
        project / f.DOWNLOAD_RECEIPT_RELATIVE,
        f._canonical_bytes(receipt),
    )
    return _SyntheticProject(project, payloads, receipt)


def _assert_self_hash(value: dict[str, object]) -> None:
    body = dict(value)
    claimed = body.pop("self_sha256")
    assert claimed == f._semantic_hash(body)


def test_production_bindings_match_final_qualifier_and_acquisition() -> None:
    assert hashlib.sha256(
        (PROJECT_ROOT / f.QUALIFIER_RELATIVE).read_bytes()
    ).hexdigest() == f.EXPECTED_QUALIFIER_SHA256
    assert hashlib.sha256(
        (PROJECT_ROOT / f.TEST_RELATIVE).read_bytes()
    ).hexdigest() == f.EXPECTED_TEST_SHA256
    assert (
        q.SOURCE_ROOT.relative_to(q.PROJECT_ROOT)
        == f.SOURCE_ROOT_RELATIVE
        == a.SOURCE_ROOT_RELATIVE
    )
    assert f.FILE_ORDER == a.FILE_ORDER
    assert {
        file_name: (
            binding.expected_size_bytes,
            binding.expected_git_blob_sha1,
        )
        for file_name, binding in f.FORMAL_FILES.items()
    } == {
        file_name: (
            binding.expected_size_bytes,
            binding.expected_git_blob_sha1,
        )
        for file_name, binding in a.FORMAL_FILES.items()
    } == {
        file_name: (
            binding.size_bytes,
            binding.git_blob_sha1,
        )
        for file_name, binding in q.FORMAL_CONTRACT.files.items()
    }


def test_synthetic_opaque_sources_build_exact_qualifier_compatible_freeze(
    synthetic_project: _SyntheticProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze = dict(f.build_qualification_freeze(synthetic_project.root))

    assert set(freeze) == {
        "download_authorization_self_sha256",
        "qualifier_sha256",
        "schema",
        "self_sha256",
        "source_custody_self_sha256",
        "source_sha256_by_file",
        "status",
        "study_design_self_sha256",
        "study_id",
        "test_sha256",
    }
    assert freeze["schema"] == "mmqa_p1_source_qualification_freeze_v1"
    assert freeze["status"] == "frozen_before_unique_formal_qualification"
    assert freeze["source_sha256_by_file"] == {
        file_name: hashlib.sha256(raw).hexdigest()
        for file_name, raw in sorted(synthetic_project.payloads.items())
    }
    serialized = json.dumps(freeze, sort_keys=True)
    assert "NOT_A_GZIP_OR_JSONL_SECRET" not in serialized
    _assert_self_hash(freeze)

    freeze_path = synthetic_project.root / f.FREEZE_RELATIVE
    assert json.loads(freeze_path.read_text("ascii")) == freeze
    assert stat.S_IMODE(freeze_path.stat().st_mode) == 0o600

    monkeypatch.setattr(q, "FREEZE_PATH", freeze_path)
    monkeypatch.setattr(
        q, "QUALIFIER_PATH", synthetic_project.root / f.QUALIFIER_RELATIVE
    )
    monkeypatch.setattr(q, "TEST_PATH", synthetic_project.root / f.TEST_RELATIVE)
    loaded, claimed = q._load_and_verify_freeze()
    assert loaded == freeze
    assert claimed == freeze["self_sha256"]


def test_existing_freeze_rejects_before_any_second_source_read(
    synthetic_project: _SyntheticProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    f.build_qualification_freeze(synthetic_project.root)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a consumed freeze must not reopen source bytes")

    monkeypatch.setattr(f, "_verify_sources", forbidden)
    with pytest.raises(
        f.MMQAP1SourceQualificationFreezeError,
        match="already consumed",
    ):
        f.build_qualification_freeze(synthetic_project.root)


def test_receipt_contract_drift_rejects_before_source_open(
    synthetic_project: _SyntheticProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = copy.deepcopy(synthetic_project.receipt)
    body["network_attempt_count"] = 5
    _replace_receipt(synthetic_project, body)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid aggregate receipt must precede source open")

    monkeypatch.setattr(f, "_verify_sources", forbidden)
    with pytest.raises(
        f.MMQAP1SourceQualificationFreezeError,
        match="receipt binding drifted",
    ):
        f.build_qualification_freeze(synthetic_project.root)
    assert not (synthetic_project.root / f.FREEZE_RELATIVE).exists()


def test_receipt_unknown_content_field_is_rejected(
    synthetic_project: _SyntheticProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = copy.deepcopy(synthetic_project.receipt)
    body["source_content"] = "SECRET_SOURCE_CONTENT_MUST_NOT_ENTER_FREEZE"
    _replace_receipt(synthetic_project, body)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("non-aggregate receipt must precede source open")

    monkeypatch.setattr(f, "_verify_sources", forbidden)
    with pytest.raises(
        f.MMQAP1SourceQualificationFreezeError,
        match="receipt shape drifted",
    ):
        f.build_qualification_freeze(synthetic_project.root)


def test_receipt_must_be_mode_0600(
    synthetic_project: _SyntheticProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    os.chmod(
        synthetic_project.root / f.DOWNLOAD_RECEIPT_RELATIVE,
        0o644,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unsafe receipt mode must precede source open")

    monkeypatch.setattr(f, "_verify_sources", forbidden)
    with pytest.raises(
        f.MMQAP1SourceQualificationFreezeError,
        match="receipt mode drifted",
    ):
        f.build_qualification_freeze(synthetic_project.root)


def test_source_sha256_must_match_aggregate_receipt(
    synthetic_project: _SyntheticProject,
) -> None:
    body = copy.deepcopy(synthetic_project.receipt)
    rows = body["files"]
    assert isinstance(rows, list)
    first = rows[0]
    assert isinstance(first, dict)
    first["sha256"] = "0" * 64
    _replace_receipt(synthetic_project, body)

    with pytest.raises(
        f.MMQAP1SourceQualificationFreezeError,
        match="SHA256 identity drifted",
    ):
        f.build_qualification_freeze(synthetic_project.root)
    assert not (synthetic_project.root / f.FREEZE_RELATIVE).exists()


def test_source_git_blob_identity_must_match(
    synthetic_project: _SyntheticProject,
) -> None:
    first_name = f.FILE_ORDER[0]
    path = synthetic_project.root / f.SOURCE_ROOT_RELATIVE / first_name
    raw = bytearray(path.read_bytes())
    raw[0] ^= 1
    _write_private(path, bytes(raw))

    with pytest.raises(
        f.MMQAP1SourceQualificationFreezeError,
        match="Git-blob identity drifted",
    ):
        f.build_qualification_freeze(synthetic_project.root)


@pytest.mark.parametrize("mode", [0o400, 0o640, 0o644])
def test_source_file_requires_exact_mode_0600(
    synthetic_project: _SyntheticProject,
    mode: int,
) -> None:
    first = (
        synthetic_project.root
        / f.SOURCE_ROOT_RELATIVE
        / f.FILE_ORDER[0]
    )
    os.chmod(first, mode)
    with pytest.raises(
        f.MMQAP1SourceQualificationFreezeError,
        match="private unique regular",
    ):
        f.build_qualification_freeze(synthetic_project.root)


def test_source_file_requires_unique_inode(
    synthetic_project: _SyntheticProject,
) -> None:
    first = (
        synthetic_project.root
        / f.SOURCE_ROOT_RELATIVE
        / f.FILE_ORDER[0]
    )
    os.link(first, synthetic_project.root / "forbidden_second_link")
    with pytest.raises(
        f.MMQAP1SourceQualificationFreezeError,
        match="private unique regular",
    ):
        f.build_qualification_freeze(synthetic_project.root)


def test_source_registry_rejects_extra_file(
    synthetic_project: _SyntheticProject,
) -> None:
    _write_private(
        synthetic_project.root
        / f.SOURCE_ROOT_RELATIVE
        / "MMQA_test.jsonl.gz",
        b"FORBIDDEN_TEST_SOURCE",
    )
    with pytest.raises(
        f.MMQAP1SourceQualificationFreezeError,
        match="file set drifted",
    ):
        f.build_qualification_freeze(synthetic_project.root)


def test_qualifier_hash_drift_rejects_before_receipt_or_source_open(
    synthetic_project: _SyntheticProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualifier = synthetic_project.root / f.QUALIFIER_RELATIVE
    qualifier.write_bytes(qualifier.read_bytes() + b"DRIFT")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("implementation drift must precede source open")

    monkeypatch.setattr(f, "_load_download_receipt", forbidden)
    monkeypatch.setattr(f, "_verify_sources", forbidden)
    with pytest.raises(
        f.MMQAP1SourceQualificationFreezeError,
        match="qualifier SHA256 drifted",
    ):
        f.build_qualification_freeze(synthetic_project.root)


def test_manifest_drift_rejects_before_implementation_or_source_open(
    synthetic_project: _SyntheticProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custody = synthetic_project.root / f.CUSTODY_RELATIVE
    custody_value = json.loads(custody.read_text("ascii"))
    custody_value["status"] = "DRIFTED"
    custody.write_text(json.dumps(custody_value), encoding="ascii")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("manifest drift must precede source open")

    monkeypatch.setattr(f, "_verify_implementation", forbidden)
    monkeypatch.setattr(f, "_verify_sources", forbidden)
    with pytest.raises(
        f.MMQAP1SourceQualificationFreezeError,
        match="semantic hash drifted",
    ):
        f.build_qualification_freeze(synthetic_project.root)


def test_cli_emits_only_aggregate_freeze(
    synthetic_project: _SyntheticProject,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert f._main(["--project", str(synthetic_project.root)]) == 0
    captured = capsys.readouterr()
    freeze = json.loads(captured.out)
    assert freeze["schema"] == "mmqa_p1_source_qualification_freeze_v1"
    assert "NOT_A_GZIP_OR_JSONL_SECRET" not in captured.out
    assert captured.err == ""
