from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from assumption_agent.benchmarks import feverous_p6_e2_acquisition_v1 as acquisition
from assumption_agent.benchmarks import feverous_p6_e2_formal_acquisition_v2 as subject


ROLLOVER_SHA = "1" * 64
LOADER_SHA = subject.rollover.TRAIN_LOADER_QUALIFICATION_SHA256
FREEZE_SHA = "2" * 64
IDENTITY_SHA = "3" * 64
SECRET = b"S" * 32


def _epoch() -> dict[str, object]:
    return {
        "source_epoch_rollover_sha256": ROLLOVER_SHA,
        "real_train_loader_qualification": {
            "qualification_sha256": LOADER_SHA,
        },
    }


def _patch_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subject.rollover,
        "verify_rollover_manifest",
        lambda _project, **_kwargs: _epoch(),
    )


def _write_private_payloads(project: Path) -> None:
    subject._exclusive_json(project / subject.CORPUS_RELATIVE, {"corpus": True})
    for relative in subject.VIEW_RELATIVES.values():
        subject._exclusive_json(project / relative, {"view": relative.name})
    for relative in subject.LABEL_RELATIVES.values():
        subject._exclusive_json(project / relative, {"labels": relative.name})


def _source_receipts() -> SimpleNamespace:
    return SimpleNamespace(
        annotation_receipt={"annotation": "aggregate"},
        database_receipt={"database": "aggregate"},
        selected_lookup_receipt={"lookup": "aggregate"},
    )


def _persist_valid_envelope(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    _patch_epoch(monkeypatch)
    subject._create_private_successor_root(project)
    _write_private_payloads(project)
    subject._exclusive_bytes(project / subject.SECRET_RELATIVE, SECRET)
    receipt = subject._public_receipt(
        project=project,
        source_epoch_rollover_sha256=ROLLOVER_SHA,
        train_loader_qualification_sha256=LOADER_SHA,
        implementation_freeze_sha256=FREEZE_SHA,
        equivalence_qualification_sha256=IDENTITY_SHA,
        secret=SECRET,
        source=_source_receipts(),
        adapter_receipt={"adapter": "aggregate"},
        selection_stats={"selection": "aggregate"},
        corpus_stats={"corpus": "aggregate"},
    )
    subject._exclusive_json(project / subject.RECEIPT_RELATIVE, receipt)
    marker_body = {
        "schema": f"{subject.VERSION}_one_shot_marker",
        "version": subject.VERSION,
        "source_epoch": "feverous_p6_e2_formal_v2",
        "source_epoch_rollover_sha256": ROLLOVER_SHA,
        "train_loader_qualification_sha256": LOADER_SHA,
        "implementation_freeze_sha256": FREEZE_SHA,
        "identity_full_compile_equivalence_qualification_sha256": IDENTITY_SHA,
        "source_split": "TRAIN",
        "retry_replay_or_resample_authorized": False,
    }
    subject._exclusive_json(
        project / subject.MARKER_RELATIVE,
        {**marker_body, "marker_sha256": subject._semantic_hash(marker_body)},
    )
    monkeypatch.setattr(subject.formal_source, "verify_annotation_receipt", lambda _r: None)
    monkeypatch.setattr(
        subject.formal_source,
        "require_formal_database_page_stream_receipt",
        lambda _r: None,
    )
    monkeypatch.setattr(
        subject.formal_source, "verify_selected_page_lookup_receipt", lambda _r: None
    )
    monkeypatch.setattr(subject.source_adapter, "verify_adapter_receipt", lambda _r: None)
    monkeypatch.setattr(
        subject.acquisition, "verify_formal_corpus_acquisition", lambda _r: None
    )
    return receipt


def test_successor_paths_are_disjoint_from_terminal_predecessor() -> None:
    assert subject.FORMAL_ROOT_RELATIVE == Path("artifacts/feverous_p6_e2_formal_v2")
    assert subject.ROOT_RELATIVE.is_relative_to(subject.FORMAL_ROOT_RELATIVE)
    assert not subject.ROOT_RELATIVE.is_relative_to(
        subject.rollover.PREDECESSOR_FORMAL_ROOT_RELATIVE
    )
    assert all(
        relative.is_relative_to(subject.FORMAL_ROOT_RELATIVE)
        for relative in (
            subject.RECEIPT_RELATIVE,
            subject.MARKER_RELATIVE,
            subject.FAILURE_RELATIVE,
            subject.SECRET_RELATIVE,
            subject.CORPUS_RELATIVE,
            *subject.VIEW_RELATIVES.values(),
            *subject.LABEL_RELATIVES.values(),
        )
    )


def test_rollover_or_loader_drift_fails_before_v2_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_epoch(monkeypatch)
    with pytest.raises(subject.FeverousFormalAcquisitionError, match="qualification"):
        subject.perform_formal_acquisition_once(
            project=tmp_path,
            source_epoch_rollover_sha256=ROLLOVER_SHA,
            train_loader_qualification_sha256="9" * 64,
            implementation_freeze_sha256=FREEZE_SHA,
            identity_full_compile_equivalence_qualification_sha256=IDENTITY_SHA,
        )
    assert not os.path.lexists(tmp_path / subject.FORMAL_ROOT_RELATIVE)


def test_marker_first_failure_is_terminal_and_retry_cannot_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_epoch(monkeypatch)
    monkeypatch.setattr(
        subject.secrets,
        "token_bytes",
        lambda _size: (_ for _ in ()).throw(RuntimeError("secret generator failed")),
    )
    arguments = {
        "project": tmp_path,
        "source_epoch_rollover_sha256": ROLLOVER_SHA,
        "train_loader_qualification_sha256": LOADER_SHA,
        "implementation_freeze_sha256": FREEZE_SHA,
        "identity_full_compile_equivalence_qualification_sha256": IDENTITY_SHA,
    }
    with pytest.raises(RuntimeError, match="secret generator failed"):
        subject.perform_formal_acquisition_once(**arguments)
    paths = subject.acquisition_paths(tmp_path)
    marker = json.loads(paths.marker.read_text("ascii"))
    failure_before = paths.failure.read_bytes()
    assert marker["source_epoch"] == "feverous_p6_e2_formal_v2"
    assert marker["source_epoch_rollover_sha256"] == ROLLOVER_SHA
    assert marker["train_loader_qualification_sha256"] == LOADER_SHA
    assert marker["implementation_freeze_sha256"] == FREEZE_SHA
    assert paths.failure.is_file()
    assert not paths.secret.exists()
    assert not paths.receipt.exists()

    with pytest.raises(subject.FeverousFormalAcquisitionError, match="already exists"):
        subject.perform_formal_acquisition_once(**arguments)
    assert paths.failure.read_bytes() == failure_before


def test_valid_envelope_binds_loader_rollover_and_all_four_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _persist_valid_envelope(tmp_path, monkeypatch)
    observed = subject.verify_acquisition_envelope(tmp_path)
    assert observed == receipt
    assert observed["source_epoch"] == "feverous_p6_e2_formal_v2"
    assert observed["source_epoch_rollover_sha256"] == ROLLOVER_SHA
    assert observed["train_loader_qualification_sha256"] == LOADER_SHA
    assert observed["block_counts"] == acquisition.BLOCK_COUNTS
    assert observed["all_blocks_one_acquisition"] is True
    assert observed["action_retrieval_utility_or_evaluator_calls"] == 0
    assert observed["F_search_gold_pack_created"] is False
    assert not (tmp_path / subject.F_SEARCH_LABEL_RELATIVE).exists()


def test_rehashed_loader_binding_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _persist_valid_envelope(tmp_path, monkeypatch)
    path = tmp_path / subject.RECEIPT_RELATIVE
    receipt = json.loads(path.read_text("ascii"))
    receipt["train_loader_qualification_sha256"] = "8" * 64
    receipt.pop("acquisition_receipt_sha256")
    receipt["acquisition_receipt_sha256"] = subject._semantic_hash(receipt)
    path.write_bytes(subject._canonical_bytes(receipt))
    os.chmod(path, 0o600)
    with pytest.raises(subject.FeverousFormalAcquisitionError, match="receipt drifted"):
        subject.verify_acquisition_envelope(tmp_path)


def test_public_receipt_never_contains_secret_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _persist_valid_envelope(tmp_path, monkeypatch)
    raw = (tmp_path / subject.RECEIPT_RELATIVE).read_bytes()
    assert SECRET not in raw
    assert SECRET.hex().encode("ascii") not in raw
    assert receipt["selection_secret_sha256"] == hashlib.sha256(SECRET).hexdigest()
    assert receipt["selection_secret_persisted_publicly"] is False
